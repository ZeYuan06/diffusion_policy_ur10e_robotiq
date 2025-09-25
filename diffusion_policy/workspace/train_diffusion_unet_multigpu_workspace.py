if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class OriginalWorkspaceCheckpointCallback(pl.Callback):
    """
    Custom callback to replicate original workspace checkpoint behavior.
    
    This callback handles:
    1. Environment rollout evaluation
    2. Diffusion sampling evaluation  
    3. JSON logging of metrics
    4. Checkpoint saving (including TopK checkpoints)
    
    The callback ensures proper execution order: rollout -> sampling -> checkpoint saving
    This guarantees that metrics like 'test_mean_score' are available before checkpoint evaluation.
    """
    
    def __init__(self, workspace, enable_validation_during_training=False):
        super().__init__()
        self.workspace = workspace
        self.topk_manager = None
        self.enable_validation_during_training = enable_validation_during_training
        
    def setup(self, trainer, pl_module, stage):
        """Initialize TopK manager on main process only"""
        if trainer.is_global_zero and self.enable_validation_during_training:
            self.topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.workspace.output_dir, 'checkpoints'),
                **self.workspace.cfg.checkpoint.topk
            )
    
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        
        cfg = self.workspace.cfg
        current_epoch = trainer.current_epoch

        self._handle_checkpoint_saving_only(trainer, cfg, current_epoch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handle rollout, sampling, and checkpoint saving following original code logic"""
        if not trainer.is_global_zero:
            return
            
        cfg = self.workspace.cfg
        current_epoch = trainer.current_epoch
        
        # If validation during training is disabled, only save checkpoints
        if not self.enable_validation_during_training:
            # self._handle_checkpoint_saving_only(trainer, cfg, current_epoch)
            return
        
        # Original full validation logic
        # ===== 1. Execute rollout and sampling to generate metrics =====
        step_log = self._run_evaluation_tasks(trainer, pl_module, cfg, current_epoch)
        
        # ===== 2. Log metrics to JSON logger =====
        self._log_to_json(trainer, current_epoch, step_log, pl_module)
        
        # ===== 3. Handle checkpoint saving =====
        self._handle_checkpoint_saving(trainer, cfg, current_epoch, step_log)
    
    def _run_evaluation_tasks(self, trainer, pl_module, cfg, current_epoch):
        """Run environment rollout and diffusion sampling tasks"""
        step_log = {}
        
        # Select policy to evaluate
        policy = self._get_evaluation_policy(pl_module, cfg)
        policy.eval()
        
        # Run environment rollout
        if self._should_run_rollout(current_epoch, cfg):
            step_log.update(self._run_environment_rollout(pl_module, policy, current_epoch))
        
        # Run diffusion sampling
        if self._should_run_sampling(current_epoch, cfg):
            step_log.update(self._run_diffusion_sampling(pl_module, policy, current_epoch))
        
        policy.train()
        return step_log
    
    def _get_evaluation_policy(self, pl_module, cfg):
        """Get the policy model for evaluation (EMA or regular model)"""
        if cfg.training.use_ema and hasattr(pl_module, 'ema_model') and getattr(pl_module, 'ema_model', None) is not None:
            policy = getattr(pl_module, 'ema_model', None)
        else:
            policy = getattr(pl_module, 'model', None)
        
        assert policy is not None, "Policy should be available for evaluation"
        return policy
    
    def _should_run_rollout(self, current_epoch, cfg):
        """Check if we should run environment rollout this epoch"""
        return ((current_epoch + 1) % cfg.training.rollout_every) == 0
    
    def _should_run_sampling(self, current_epoch, cfg):
        """Check if we should run diffusion sampling this epoch"""
        return ((current_epoch + 1) % cfg.training.sample_every) == 0
    
    def _run_environment_rollout(self, pl_module, policy, current_epoch, current_step=None):
        """Run environment rollout and return logged metrics"""
        step_log = {}
        
        # Check if env_runner exists in pl_module (where it's actually created)
        if hasattr(pl_module, 'env_runner') and pl_module.env_runner is not None:
            print(f"Running environment rollout at epoch {current_epoch}")
            was_training = policy.training
            policy.eval()

            with torch.no_grad():
                runner_log = pl_module.env_runner.run(policy, current_step=current_step)
                step_log.update(runner_log)

            policy.train(was_training)
            
            # Log to wandb
            for key, value in runner_log.items():
                pl_module.log(f'env/{key}', value, step=current_epoch if current_epoch else 0, sync_dist=True)
        else:
            print("Warning: env_runner is not set, skipping environment evaluation.")
        
        return step_log
    
    def _run_diffusion_sampling(self, pl_module, policy, current_epoch):
        """Run diffusion sampling and return logged metrics"""
        step_log = {}
        
        train_sampling_batch = getattr(pl_module, 'train_sampling_batch', None)
        if train_sampling_batch is not None:
            print(f"Running diffusion sampling at epoch {current_epoch}")
            with torch.no_grad():
                # Sample trajectory from training set, and evaluate difference
                obs_dict = dict_apply(
                    train_sampling_batch['obs'], 
                    lambda x: x.to(pl_module.device)
                )
                gt_action = train_sampling_batch['action'].to(pl_module.device)
                
                result = policy.predict_action(obs_dict)
                pred_action = result['action_pred']
                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                
                step_log['train_action_mse_error'] = mse.item()
                pl_module.log('train_action_mse_error', mse, on_epoch=True, sync_dist=True)
                
                # Clean up memory
                del obs_dict, gt_action, result, pred_action, mse
        else:
            print("Warning: train_sampling_batch is not set, skipping sampling evaluation.")
        
        return step_log
    
    def _log_to_json(self, trainer, current_epoch, step_log, pl_module):
        """Log metrics to JSON logger"""
        # Check if json_logger exists in pl_module (where it's actually created)
        if hasattr(pl_module, 'json_logger') and pl_module.json_logger is not None:
            final_log = {
                'global_step': trainer.global_step,
                'epoch': current_epoch,
            }
            
            # Add training metrics
            if hasattr(trainer, 'logged_metrics') and trainer.logged_metrics:
                self._add_trainer_metrics(final_log, trainer.logged_metrics)
            
            # Add evaluation metrics
            final_log.update(step_log)
            
            # Log to json file
            pl_module.json_logger.log(final_log)
    
    def _add_trainer_metrics(self, final_log, logged_metrics):
        """Add trainer logged metrics to final log"""
        for key, value in logged_metrics.items():
            if isinstance(value, torch.Tensor):
                final_log[key] = float(value.item())
    
    def _handle_checkpoint_saving(self, trainer, cfg, current_epoch, step_log):
        """Handle checkpoint saving logic"""
        if not self._should_save_checkpoint(current_epoch, cfg):
            return
        
        # Save standard checkpoints
        self._save_standard_checkpoints(cfg)
        
        # Handle TopK checkpoint saving
        self._save_topk_checkpoint(trainer, step_log)
    
    def _handle_checkpoint_saving_only(self, trainer, cfg, current_epoch):
        """Handle checkpoint saving only without validation/rollout/sampling"""
        if not self._should_save_checkpoint(current_epoch, cfg):
            return
        
        print(f"Saving checkpoint at epoch {current_epoch} (training-only mode)")
        
        # Save standard checkpoints without TopK strategy
        if cfg.checkpoint.save_last_ckpt:
            self.workspace.save_checkpoint()
        
        if cfg.checkpoint.save_last_snapshot:
            self.workspace.save_snapshot()
        
        # In training-only mode, save all checkpoints with epoch number
        epoch_ckpt_path = os.path.join(
            self.workspace.output_dir, 
            'checkpoints', 
            f'epoch_{current_epoch:04d}.ckpt'
        )
        self.workspace.save_checkpoint(path=epoch_ckpt_path)
        print(f"Saved epoch checkpoint: {epoch_ckpt_path}")
    
    def _should_save_checkpoint(self, current_epoch, cfg):
        """Check if we should save checkpoint this epoch"""
        return ((current_epoch + 1) % cfg.training.checkpoint_every) == 0
    
    def _save_standard_checkpoints(self, cfg):
        """Save last checkpoint and snapshot"""
        if cfg.checkpoint.save_last_ckpt:
            self.workspace.save_checkpoint()
        
        if cfg.checkpoint.save_last_snapshot:
            self.workspace.save_snapshot()
    
    def _save_topk_checkpoint(self, trainer, step_log):
        """Save TopK checkpoint if metrics are available"""
        if self.topk_manager is None:
            return
        
        # Collect all available metrics
        metric_dict = self._collect_metrics(trainer, step_log)
        print(f"Available metrics for TopK: {list(metric_dict.keys())}")
        
        try:
            # Check if required metric exists and save checkpoint
            monitor_key = getattr(self.topk_manager, 'monitor_key', None)
            if monitor_key and monitor_key in metric_dict:
                topk_ckpt_path = self.topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.workspace.save_checkpoint(path=topk_ckpt_path)
                    print(f"Saved TopK checkpoint: {topk_ckpt_path}")
            else:
                print(f"Warning: Monitor key '{monitor_key}' not found in metrics.")
                print(f"Available metrics: {list(metric_dict.keys())}")
                # Fallback to val_loss if available
                self._try_fallback_metric(metric_dict)
        except Exception as e:
            print(f"Error saving TopK checkpoint: {e}")
    
    def _collect_metrics(self, trainer, step_log):
        """Collect all metrics from trainer and step_log"""
        metric_dict = {}
        
        # Add trainer logged metrics
        if hasattr(trainer, 'logged_metrics') and trainer.logged_metrics:
            for key, value in trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value.item()
        
        # Add step_log metrics
        for key, value in step_log.items():
            if key not in metric_dict:
                metric_dict[key.replace('/', '_')] = value
        
        return metric_dict
    
    def _try_fallback_metric(self, metric_dict):
        """Try to use fallback metric (val_loss) for checkpoint saving"""
        if 'val_loss' in metric_dict and self.topk_manager is not None:
            print("Using val_loss as fallback metric for TopK checkpoint")
            try:
                # Temporarily modify topk_manager settings
                original_key = getattr(self.topk_manager, 'monitor_key', None)
                original_mode = getattr(self.topk_manager, 'mode', None)
                
                if hasattr(self.topk_manager, 'monitor_key'):
                    self.topk_manager.monitor_key = 'val_loss'
                if hasattr(self.topk_manager, 'mode'):
                    self.topk_manager.mode = 'min'  # val_loss should be minimized
                
                topk_ckpt_path = self.topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.workspace.save_checkpoint(path=topk_ckpt_path)
                    print(f"Saved TopK checkpoint using val_loss: {topk_ckpt_path}")
                
                # Restore original settings
                if original_key is not None and hasattr(self.topk_manager, 'monitor_key'):
                    self.topk_manager.monitor_key = original_key
                if original_mode is not None and hasattr(self.topk_manager, 'mode'):
                    self.topk_manager.mode = original_mode
            except Exception as e:
                print(f"Error using fallback metric: {e}")


class DiffusionPolicyMultiGpuLightningModule(pl.LightningModule):
    """Lightning Module for multi-GPU diffusion policy training"""
    
    def __init__(self, cfg: OmegaConf, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.output_dir = output_dir

        # Initialize model-related attributes (loaded lazily)
        self.model = None
        self.ema_model = None
        self.ema = None
        
        # Training utilities
        self.train_sampling_batch = None
        
        # Main process only attributes
        self.env_runner = None
        self.json_logger = None
        
    def configure_model(self, normalizer=None):
        """Configure model and EMA model, optionally set normalizer"""
        if self.model is None:
            # Create the model only when needed
            self.model = hydra.utils.instantiate(self.cfg.policy)
            
            # EMA model
            if self.cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
        
        # Set normalizer (if provided)
        if normalizer is not None:
            self.model.set_normalizer(normalizer)
            if self.cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)
                # Initialize EMA updater
                self.ema = hydra.utils.instantiate(
                    self.cfg.ema,
                    model=self.ema_model
                )
        
    def setup(self, stage=None):
        """Lightning calls this method on each process"""
        if stage == 'fit':
            # Configure model if not already done
            if self.model is None:
                self.configure_model()
                
            # Setup main process only components
            if self.trainer.is_global_zero:
                self._setup_env_runner()
                self._setup_json_logger()
    
    def _setup_env_runner(self):
        """Setup environment runner for evaluation (main process only)"""
        self.env_runner = hydra.utils.instantiate(
            self.cfg.task.env_runner,
            output_dir=self.output_dir
        )
    
    def _setup_json_logger(self):
        """Setup JSON logger for training metrics (main process only)"""
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.json_logger = JsonLogger(log_path)
        self.json_logger.start()
    
    def configure_optimizers(self):
        # Ensure the model is already configured
        assert self.model is not None, "Model should be configured before optimizer setup"
        
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer,
            params=self.model.parameters()
        )        # calculate training steps
                        
        try:
            num_training_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            print("Warning: `trainer.estimated_stepping_batches` not available. Falling back to manual calculation.")
            world_size = self.trainer.world_size if self.trainer.world_size > 0 else 1
            train_dataloader_len = len(self.trainer.datamodule.train_dataloader())
            num_training_steps = (
                train_dataloader_len * self.cfg.training.num_epochs
            ) // (self.cfg.training.gradient_accumulate_every * world_size)
        
        lr_scheduler = get_scheduler(
            self.cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1
        )
        
        return [optimizer], [{
                "scheduler": lr_scheduler,
                "interval": "step",  # update every step
                "frequency": 1,
            }]
    
    def training_step(self, batch, batch_idx):
        # Ensure the model is already configured
        assert self.model is not None, "Model should be configured before training"
        
        # Save first batch for sampling
        self._save_sampling_batch(batch)
            
        # Compute loss
        raw_loss = self.model.compute_loss(batch)
        loss = raw_loss / self.cfg.training.gradient_accumulate_every
        
        # Update EMA (only on main process)
        self._update_ema()
        
        # Log metrics
        self._log_training_metrics(raw_loss, batch_idx)
        
        return loss
    
    def _save_sampling_batch(self, batch):
        """Save first batch for sampling evaluation"""
        if self.train_sampling_batch is None:
            self.train_sampling_batch = {
                'obs': dict_apply(batch['obs'], lambda x: x.cpu()),
                'action': batch['action'].cpu()
            }
    
    def _update_ema(self):
        """Update EMA model on main process only"""
        if self.cfg.training.use_ema and self.trainer.is_global_zero and self.ema is not None:
            self.ema.step(self.model)
    
    def _log_training_metrics(self, raw_loss, batch_idx):
        """Log training metrics to wandb and json logger"""
        # Log to wandb
        self.log('train_loss', raw_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, sync_dist=False)
        
        # Log to json logger (only on main process)
        if (self.trainer.is_global_zero and 
            hasattr(self, 'json_logger') and 
            self.json_logger is not None):
            step_log = {
                'train_loss': raw_loss.item(),
                'global_step': self.global_step,
                'epoch': self.current_epoch,
                'lr': self.trainer.optimizers[0].param_groups[0]['lr']
            }
            # Only log intermediate steps, final step is logged in callback
            if not self._is_last_batch_in_epoch(batch_idx):
                self.json_logger.log(step_log)
    
    def _is_last_batch_in_epoch(self, batch_idx):
        """Check if this is the last batch in the current epoch"""
        try:
            # estimate based on dataloader length
            if hasattr(self.trainer, 'num_training_batches'):
                return batch_idx >= (self.trainer.num_training_batches - 1)
            return False
        except:
            raise ValueError("Please set `num_training_batches` in the config if the dataloader does not support __len__.")
    
    def validation_step(self, batch, batch_idx):
        assert self.model is not None, "Model should be configured before validation"
        loss = self.model.compute_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Simplified validation epoch end - rollout and sampling logic moved to callback"""
        if not self.trainer.is_global_zero:
            return
        
        # The rollout and sampling logic is now handled by the OriginalWorkspaceCheckpointCallback
        # This ensures proper execution order: rollout -> sampling -> checkpoint saving
        print(f"Validation epoch {self.current_epoch} completed")
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # this ensures proper cleanup and follows original code structure
        pass
    
    def on_fit_end(self):
        """Called when training ends"""
        # close json logger properly
        if self.trainer.is_global_zero and self.json_logger is not None:
            self.json_logger.stop()

class DiffusionMultiGpuDataModule(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__()
        self.cfg = cfg
        self.output_dir = output_dir

        # Initialize all dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.normalizer = None
        
    def setup(self, stage=None):
        """Setup data module based on stage: 'fit' for training, 'validate' for evaluation"""
        print(f"Setting up data module for stage: {stage}")
        
        # Set up validation mask save path for consistent train/val splits
        mask_params = self._setup_validation_mask_path(stage=stage)
        
        # Create the full dataset with validation mask configuration
        full_dataset = hydra.utils.instantiate(
            self.cfg.task.dataset,
            **mask_params
        )
        
        if stage == "fit" or stage is None:
            # Training stage: setup training data and save validation split
            self._setup_for_training(full_dataset)
        elif stage in ["validate", "test"]:
            # Evaluation stage: load saved validation split
            self._setup_for_evaluation(full_dataset)
        else:
            # Default: setup both datasets
            self._setup_default(full_dataset)
            
    def _setup_validation_mask_path(self, stage=None):
        """Configure validation mask save path safely for multi-process training"""
        mask_params = {}
        if not self.output_dir:
            return mask_params
            
        val_mask_path = os.path.join(self.output_dir, 'val_mask.pkl')
        mask_params['val_mask_save_path'] = val_mask_path
        
        if stage == 'fit':
            # Training stage: main process creates, others wait and load
            if self._is_main_process():
                if os.path.exists(val_mask_path):
                    mask_params['load_existing_val_mask'] = True
                    print(f"Main process loading existing validation mask: {val_mask_path}")
                else:
                    mask_params['load_existing_val_mask'] = False
                    print(f"Main process creating new validation mask: {val_mask_path}")
            else:
                self._wait_for_validation_mask(val_mask_path)
                mask_params['load_existing_val_mask'] = True
                print(f"Worker process loading validation mask: {val_mask_path}")
        elif stage in ['validate', 'test']:
            # Evaluation stage: all processes load existing mask
            mask_params['load_existing_val_mask'] = True
            print(f"Loading existing validation mask: {val_mask_path}")
        else:
            raise ValueError(f"Unknown stage: {stage}")
            
        return mask_params
    
    def _is_main_process(self):
        """Check if this is the main process"""
        return int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    def _wait_for_validation_mask(self, val_mask_path, timeout=300):
        """Wait for main process to create validation mask file"""
        import time
        
        start_time = time.time()
        while not os.path.exists(val_mask_path):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for validation mask: {val_mask_path}")
            
            print("Waiting for main process to create validation mask...")
            time.sleep(2)
        
        # Additional wait to ensure file write is complete
        time.sleep(1)

    def _setup_for_training(self, full_dataset):
        """Setup for training: use training data only, save validation split"""
        print("Setting up for training stage...")
        
        # Get training-only dataset (excludes validation episodes)
        self.train_dataset = full_dataset.get_train_only_dataset()
        # Also get validation dataset to save the split
        self.val_dataset = full_dataset.get_validation_dataset()

        # Compute normalizer from TRAINING DATA ONLY
        self.normalizer = self.train_dataset.get_normalizer(**self.cfg.task.dataset.get('normalizer', {}))
        
        # Save normalizer for evaluation stage
        self._save_normalizer()
        
        print(f"Training setup complete:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)} (saved for evaluation)")
        print(f"  Normalizer computed from training data only")
        
    def _setup_for_evaluation(self, full_dataset):
        """Setup for evaluation: load saved validation split and normalizer"""
        print("Setting up for evaluation stage...")
        
        # Get validation dataset using saved validation mask
        self.val_dataset = full_dataset.get_validation_dataset()
        
        # Try to load saved normalizer first
        if self._load_saved_normalizer():
            print("Loaded saved normalizer from training")
        else:
            print("Warning: No saved normalizer found, computing from training data")
            # Fallback: compute from training data
            train_dataset = full_dataset.get_train_only_dataset()
            self.normalizer = train_dataset.get_normalizer(**self.cfg.task.dataset.get('normalizer', {}))
        
        print(f"Evaluation setup complete:")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Using saved normalizer for consistency")

    def _setup_default(self, full_dataset):
        """Default setup: create both datasets"""
        print("Setting up default configuration...")

        self.train_dataset = full_dataset.get_train_only_dataset()
        self.val_dataset = full_dataset.get_validation_dataset()
        self.normalizer = self.train_dataset.get_normalizer(**self.cfg.task.dataset.get('normalizer', {}))
        
        print(f"Default setup complete:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
    
    def _save_normalizer(self):
        """Save normalizer for consistent evaluation"""
        if not self.output_dir or not self.normalizer:
            return
            
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        try:
            os.makedirs(os.path.dirname(normalizer_path), exist_ok=True)
            import pickle
            with open(normalizer_path, 'wb') as f:
                pickle.dump(self.normalizer, f)
            print(f"Saved normalizer to: {normalizer_path}")
        except Exception as e:
            print(f"Warning: Could not save normalizer: {e}")
    
    def _load_saved_normalizer(self):
        """Load saved normalizer from training"""
        if not self.output_dir:
            return False
            
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if not os.path.exists(normalizer_path):
            return False
        
        try:
            import pickle
            with open(normalizer_path, 'rb') as f:
                self.normalizer = pickle.load(f)
            print(f"Loaded normalizer from: {normalizer_path}")
            return True
        except Exception as e:
            print(f"Warning: Could not load saved normalizer: {e}")
            return False
        
    def train_dataloader(self):
        """Return training dataloader"""
        if self.train_dataset is None:
            raise RuntimeError("No training dataset available. Call setup('fit') first.")
        return DataLoader(self.train_dataset, **self.cfg.dataloader)
    
    def val_dataloader(self):
        """Return validation dataloader"""
        if self.val_dataset is None:
            raise RuntimeError("No validation dataset available. Call setup('fit') first.")
        return DataLoader(self.val_dataset, **self.cfg.val_dataloader)

class TrainDiffusionUnetMultiGpuWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        pl.seed_everything(cfg.training.seed, workers=True)

        # create lightning model
        self.lightning_model = DiffusionPolicyMultiGpuLightningModule(cfg, output_dir=self.output_dir)

        self.cfg = cfg
        self._current_trainer = None

    def set_trainer(self, trainer):
        """Set trainer reference for adding Lightning metadata to checkpoints"""
        self._current_trainer = trainer

    def save_checkpoint(self, path=None, tag='latest', 
                       exclude_keys=None, include_keys=None, use_thread=True):
        """Override to add PyTorch Lightning compatible metadata"""
        import dill
        import threading
        
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }
        
        # Add trainer metadata if available
        if self._current_trainer is not None:
            payload['epoch'] = self._current_trainer.current_epoch
            payload['global_step'] = self._current_trainer.global_step
            payload['pytorch-lightning_version'] = pl.__version__

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        from diffusion_policy.workspace.base_workspace import _copy_to_cpu
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
                
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        return str(path.absolute())

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # create data module with output_dir for validation mask saving
        data_module = DiffusionMultiGpuDataModule(cfg, output_dir=self.output_dir)
        data_module.setup(stage='fit')
        
        # create lightning model
        lightning_model = self.lightning_model
        
        # Use the new configure_model method to configure the model and normalizer at once
        lightning_model.configure_model(normalizer=data_module.normalizer)

        # configure wandb logger
        wandb_logger = WandbLogger(
            save_dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        
        # configure callbacks
        callbacks = []
        
        # Add original workspace checkpoint callback
        # Set enable_validation_during_training to False for training-only mode
        checkpoint_callback = OriginalWorkspaceCheckpointCallback(
            self, 
            enable_validation_during_training=False
        )
        callbacks.append(checkpoint_callback)
        
        # learning rate monitor callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # configure trainer
        trainer = pl.Trainer(
            max_epochs=cfg.training.num_epochs,
            devices=cfg.training.get('devices', 'auto'),  # auto detect GPU count
            accelerator=cfg.training.get('accelerator', 'gpu'),
            strategy=cfg.training.get('strategy', 'ddp'),  # distributed strategy
            logger=wandb_logger,
            callbacks=callbacks,
            gradient_clip_val=cfg.training.get('gradient_clip_val', None),
            accumulate_grad_batches=cfg.training.gradient_accumulate_every,
            check_val_every_n_epoch=None,    #cfg.training.checkpoint_every,  # Only validate when saving checkpoints
            enable_checkpointing=False,
            num_sanity_val_steps=0,  # Skip sanity validation steps in training-only mode
            precision=cfg.training.get('precision', 32),
            enable_progress_bar=True,
            enable_model_summary=True,
            # add debug mode support (following original code)
            fast_dev_run=cfg.training.get('debug', False),
            limit_train_batches=cfg.training.get('max_train_steps', None),
            limit_val_batches=0, #cfg.training.get('max_val_steps', None),
        )
        
        # Set trainer reference for checkpoint metadata
        self.set_trainer(trainer)
        
        # resume from checkpoint
        ckpt_path = None
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                ckpt_path = str(lastest_ckpt_path)
                print(f"Resuming from checkpoint {ckpt_path}")
        
        # start training
        trainer.fit(
            model=lightning_model,
            datamodule=data_module,
            ckpt_path=ckpt_path
        )

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetMultiGpuWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
