"""
Script to evaluate all saved checkpoints after training completion.
This script reuses the existing PyTorch Lightning workspace infrastructure
to ensure consistency and robustness.
"""

import os
import re
import glob
import torch
import hydra
from omegaconf import OmegaConf
import pandas as pd
import json
from datetime import datetime
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from diffusion_policy.workspace.train_diffusion_unet_multigpu_workspace import (
    TrainDiffusionUnetMultiGpuWorkspace,
    DiffusionMultiGpuDataModule,
    OriginalWorkspaceCheckpointCallback,
)

torch.set_float32_matmul_precision('high')

class CheckpointEvaluationCallback(OriginalWorkspaceCheckpointCallback):
    """
    Extended callback for checkpoint evaluation that forces evaluation tasks
    """

    def __init__(self, workspace, checkpoint_path, epoch_info):
        # Enable validation during evaluation
        super().__init__(workspace, enable_validation_during_training=True)
        self.checkpoint_path = checkpoint_path
        self.epoch_info = epoch_info
        self.evaluation_results = {}

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Force run all evaluation tasks for checkpoint evaluation"""
        if not trainer.is_global_zero:
            return

        cfg = self.workspace.cfg
        current_epoch = self.epoch_info.get("epoch", 0)
        current_step = self.epoch_info.get("global_step", 0)

        print(
            f"Evaluating checkpoint: {os.path.basename(self.checkpoint_path)} (epoch {current_epoch})"
        )

        # Force run all evaluation tasks regardless of epoch intervals
        step_log = {}

        # Select policy to evaluate
        policy = self._get_evaluation_policy(pl_module, cfg)
        policy.eval()

        # 1. Force run environment rollout
        print("  Running environment rollout...")
        step_log.update(
            self._run_environment_rollout(
                pl_module, policy, current_epoch, current_step=current_step
            )
        )

        # # 2. Force run diffusion sampling
        # print("  Running diffusion sampling...")
        # step_log.update(self._run_diffusion_sampling(pl_module, policy, current_epoch))

        policy.train()

        # Store results for later access
        self.evaluation_results = step_log.copy()

        # Add checkpoint metadata
        self.evaluation_results.update(
            {
                "checkpoint": os.path.basename(self.checkpoint_path),
                "checkpoint_path": self.checkpoint_path,
                **self.epoch_info,
            }
        )

        # Log to JSON logger if available
        self._log_to_json(trainer, current_epoch, step_log, pl_module)

        print(f"  Evaluation completed for {os.path.basename(self.checkpoint_path)}")


def setup_wandb_logger_for_evaluation(cfg, checkpoint_dir):
    """
    Set up wandb logger for evaluation, trying to resume the original training run
    """
    try:
        # Try to find wandb run info from the training output
        output_dir = checkpoint_dir
        wandb_dir = os.path.join(output_dir, "wandb")

        wandb_config = cfg.logging if "logging" in cfg else {}

        if os.path.exists(wandb_dir):
            # Find the latest run directory
            run_dirs = [d for d in os.listdir(wandb_dir) if d.startswith("run-")]
            if run_dirs:
                latest_run_dir = max(
                    run_dirs, key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x))
                )
                run_id = latest_run_dir.replace("run-", "").split("-", 1)[-1]

                print(f"Attempting to resume wandb run: {run_id}")

                # Try to resume the original run
                wandb_logger = WandbLogger(
                    save_dir=output_dir,
                    project=wandb_config.get("project", "diffusion_policy_debug"),
                    id=run_id,
                    resume="allow",
                    config=OmegaConf.to_container(cfg, resolve=True),
                )

                print(f"Successfully resumed wandb run: {wandb_logger.experiment.name}")
                return wandb_logger

    except Exception as e:
        print(f"Could not resume original wandb run: {e}")

    # Create new run if resuming fails
    print("Creating new wandb run for evaluation...")

    original_name = cfg.logging.get("name", "train_diffusion_unet")
    eval_name = f"{original_name}_checkpoint_eval"

    wandb_logger = WandbLogger(
        save_dir=checkpoint_dir,
        name=eval_name,
        project=cfg.logging.get("project", "diffusion_policy_debug"),
        group=cfg.logging.get("group", None),
        tags=["checkpoint_evaluation"] + cfg.logging.get("tags", []),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    return wandb_logger


def evaluate_single_checkpoint(
    workspace, data_module, ckpt_path, wandb_logger, use_wandb=True
):
    """
    Evaluate a single checkpoint using the Lightning infrastructure

    Args:
        workspace: TrainDiffusionUnetMultiGpuWorkspace instance
        data_module: DiffusionMultiGpuDataModule instance
        ckpt_path: Path to checkpoint file
        wandb_logger: WandbLogger instance
        use_wandb: Whether to use wandb logging

    Returns:
        dict: Evaluation results
    """
    try:
        # Load checkpoint to get metadata
        checkpoint = torch.load(ckpt_path)
        if "epoch" in checkpoint:
            epoch_info = {
                "epoch": checkpoint["epoch"],
                "global_step": checkpoint.get("global_step", -1),
            }
        else:
            # Extract epoch from the checkpoint filename
            match = re.search(r"epoch_(\d+)", os.path.basename(ckpt_path))
            epoch = int(match.group(1)) if match else -1
            epoch_info = {
                "epoch": epoch + 1,
                "global_step": checkpoint.get("global_step", -1),
            }

        # Create evaluation callback
        eval_callback = CheckpointEvaluationCallback(workspace, ckpt_path, epoch_info)

        # Configure trainer for evaluation (single step validation)
        trainer = pl.Trainer(
            max_epochs=1,
            devices="auto",  # Use single device for evaluation
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=wandb_logger if use_wandb else None,
            callbacks=[eval_callback],
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=1,  # Only need one validation step to trigger our callback
        )

        # Create lightning model
        lightning_model = workspace.lightning_model
        lightning_model.configure_model(normalizer=data_module.normalizer)

        # Load the checkpoint state
        lightning_model.load_state_dict(
            checkpoint["state_dicts"]["lightning_model"], strict=False
        )

        # Set up components needed for evaluation
        lightning_model.env_runner = hydra.utils.instantiate(
            workspace.cfg.task.env_runner, output_dir=workspace.output_dir
        )

        # Create sampling batch from validation data
        val_dataloader = data_module.val_dataloader()
        sampling_batch = next(iter(val_dataloader))
        from diffusion_policy.common.pytorch_util import dict_apply

        lightning_model.train_sampling_batch = {
            "obs": dict_apply(sampling_batch["obs"], lambda x: x.cpu()),
            "action": sampling_batch["action"].cpu(),
        }

        # Set up JSON logger
        if (
            not hasattr(lightning_model, "json_logger")
            or lightning_model.json_logger is None
            or lightning_model.json_logger.file is None
        ):
            from diffusion_policy.common.json_logger import JsonLogger

            log_path = os.path.join(
                workspace.output_dir,
                f"eval_logs_{os.path.basename(ckpt_path)}.json.txt",
            )
            lightning_model.json_logger = JsonLogger(log_path)
            lightning_model.json_logger.start()

        # Run validation to trigger evaluation
        trainer.validate(
            model=lightning_model,
            datamodule=data_module,
            ckpt_path=None,  # Already loaded
            verbose=False,
        )

        # Get evaluation results from callback
        results = eval_callback.evaluation_results.copy()

        # Get validation loss from trainer's logged metrics
        if hasattr(trainer, "logged_metrics") and "val_loss" in trainer.logged_metrics:
            val_loss = (
                trainer.logged_metrics["val_loss"].item()
                if torch.is_tensor(trainer.logged_metrics["val_loss"])
                else trainer.logged_metrics["val_loss"]
            )
            results["val_loss"] = val_loss

            # Log validation loss to wandb if available
            if use_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
                wandb_logger.experiment.log(
                    {
                        "eval_validation/val_loss": val_loss,
                        "eval_epoch": epoch_info["epoch"],
                        "eval_checkpoint": os.path.basename(ckpt_path),
                    },
                    step=(
                        epoch_info["global_step"]
                        if epoch_info["global_step"] >= 0
                        else None
                    ),
                )

        # Close JSON logger
        if hasattr(lightning_model, "json_logger") and lightning_model.json_logger:
            lightning_model.json_logger.stop()

        print(f"Results for {os.path.basename(ckpt_path)}:")
        for key, value in results.items():
            if isinstance(value, (int, float)) and key not in ["epoch", "global_step"]:
                print(f"  {key}: {value:.4f}")

        return results

    except Exception as e:
        print(f"Error evaluating {ckpt_path}: {e}")
        import traceback

        traceback.print_exc()
        return None


def evaluate_all_checkpoints(config_path, checkpoint_dir, output_csv, use_wandb=True):
    """
    Evaluate all checkpoints using the Lightning workspace infrastructure

    Args:
        config_path: Path to configuration file
        checkpoint_dir: Directory containing checkpoints
        output_csv: Path to save results CSV file
        use_wandb: Whether to use wandb logging
    """
    print("Loading configuration and setting up workspace...")

    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Create workspace and data module
    # output_dir = os.path.dirname(checkpoint_dir)
    workspace = TrainDiffusionUnetMultiGpuWorkspace(cfg, output_dir=checkpoint_dir)

    # Set up data module
    data_module = DiffusionMultiGpuDataModule(cfg, output_dir=checkpoint_dir)
    data_module.setup(stage="validate")

    # Set up wandb logger
    wandb_logger = None
    if use_wandb:
        try:
            wandb_logger = setup_wandb_logger_for_evaluation(cfg, checkpoint_dir)
            print(f"Wandb logging enabled: {wandb_logger.experiment.url}")
        except Exception as e:
            print(f"Failed to setup wandb: {e}")
            print("Continuing without wandb logging...")
            use_wandb = False

    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoints", "*.ckpt"))
    checkpoint_files.sort()

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Log evaluation start to wandb
    if use_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
        wandb_logger.experiment.log(
            {
                "eval_meta/total_checkpoints": len(checkpoint_files),
                "eval_meta/evaluation_start": datetime.now().isoformat(),
                "eval_meta/config_path": config_path,
                "eval_meta/checkpoint_dir": checkpoint_dir,
            },
            step=0,
        )

    results = []

    for i, ckpt_path in enumerate(checkpoint_files[-4:]):
        print(f"\n--- Evaluating checkpoint {i+1}/{len(checkpoint_files)} ---")

        result = evaluate_single_checkpoint(
            workspace, data_module, ckpt_path, wandb_logger, use_wandb
        )

        if result is not None:
            results.append(result)

            # Save intermediate results
            temp_csv = output_csv.replace(".csv", "_temp.csv")
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(temp_csv, index=False)

    # Save final results
    if results:
        df = pd.DataFrame(results)
        if "epoch" in df.columns:
            df = df.sort_values("epoch", ascending=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nFinal results saved to {output_csv}")

        # Create summary in wandb
        if use_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
            # Create summary statistics
            numeric_cols = []
            for col in df.columns:
                if col not in [
                    "checkpoint",
                    "checkpoint_path",
                ] and pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)

            summary_data = {"eval_summary/total_evaluated": len(results)}

            for col in numeric_cols:
                if df[col].notna().any():
                    summary_data.update(
                        {
                            f"eval_summary/{col}_mean": df[col].mean(),
                            f"eval_summary/{col}_std": df[col].std(),
                            f"eval_summary/{col}_min": df[col].min(),
                            f"eval_summary/{col}_max": df[col].max(),
                        }
                    )

                    # Find best checkpoint
                    if any(
                        keyword in col.lower() for keyword in ["loss", "error", "mse"]
                    ):
                        best_idx = df[col].idxmin()
                    else:
                        best_idx = df[col].idxmax()

                    summary_data.update(
                        {
                            f"eval_best/{col}_value": df.loc[best_idx, col],
                            f"eval_best/{col}_checkpoint": df.loc[
                                best_idx, "checkpoint"
                            ],
                            f"eval_best/{col}_epoch": (
                                df.loc[best_idx, "epoch"]
                                if "epoch" in df.columns
                                else -1
                            ),
                        }
                    )

            # Log summary and create table
            wandb_logger.experiment.log(summary_data, step=0)

            # Create results table
            import wandb

            table_columns = ["checkpoint", "epoch"] + [
                col for col in numeric_cols if col != "epoch"
            ]
            table_data = []

            for _, row in df.iterrows():
                table_row = [row.get("checkpoint", ""), row.get("epoch", -1)]
                for col in table_columns[2:]:
                    table_row.append(row.get(col, None))
                table_data.append(table_row)

            results_table = wandb.Table(columns=table_columns, data=table_data)
            wandb_logger.experiment.log({"eval_summary/results_table": results_table})

        # Print best checkpoints
        print("\n" + "=" * 60)
        print("BEST CHECKPOINTS BY METRIC:")
        print("=" * 60)

        metrics_to_check = ["test_mean_score", "val_loss", "train_action_mse_error"]
        for metric in metrics_to_check:
            if metric in df.columns and df[metric].notna().any():
                if metric in ["val_loss", "train_action_mse_error"]:
                    best_idx = df[metric].idxmin()
                else:
                    best_idx = df[metric].idxmax()

                print(f"\nBest checkpoint by {metric}:")
                print(f"  Checkpoint: {df.loc[best_idx, 'checkpoint']}")
                print(f"  Value: {df.loc[best_idx, metric]:.4f}")
                if "epoch" in df.columns:
                    print(f"  Epoch: {df.loc[best_idx, 'epoch']}")

        # Save detailed JSON results
        json_output = output_csv.replace(".csv", ".json")
        with open(json_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {json_output}")

        # Clean up temp file
        temp_csv = output_csv.replace(".csv", "_temp.csv")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        # Log completion to wandb
        if use_wandb and wandb_logger and hasattr(wandb_logger, "experiment"):
            wandb_logger.experiment.log(
                {
                    "eval_meta/evaluation_end": datetime.now().isoformat(),
                    "eval_meta/successful_evaluations": len(results),
                },
                step=0,
            )

            print(f"\nAll results logged to wandb: {wandb_logger.experiment.url}")

    else:
        print("No successful evaluations completed")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints using Lightning workspace"
    )
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint_dir", required=True, help="Directory containing training outputs"
    )
    parser.add_argument(
        "--output", default="evaluation_results.csv", help="Output CSV file"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} does not exist")
        return

    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist")
        return

    # Ensure output directory exists
    output = os.path.join(args.checkpoint_dir, args.output)

    print(f"Configuration: {args.config}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output file: {output}")
    print(f"Wandb logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print("-" * 50)

    evaluate_all_checkpoints(
        args.config, args.checkpoint_dir, output, use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()
