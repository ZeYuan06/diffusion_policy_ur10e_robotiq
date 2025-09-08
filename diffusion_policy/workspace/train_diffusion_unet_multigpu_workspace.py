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
    """Custom callback to replicate original workspace checkpoint behavior"""
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.topk_manager = None
        
    def setup(self, trainer, pl_module, stage):
        """Initialize TopK manager on main process only"""
        if trainer.is_global_zero:
            self.topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.workspace.output_dir, 'checkpoints'),
                **self.workspace.cfg.checkpoint.topk
            )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Handle checkpoint saving following original code logic"""
        if not trainer.is_global_zero:
            return
            
        cfg = self.workspace.cfg
        current_epoch = trainer.current_epoch
        
        # checkpoint handling (following original code frequency control)
        if (current_epoch % cfg.training.checkpoint_every) == 0:
            # save last checkpoint
            if cfg.checkpoint.save_last_ckpt:
                self.workspace.save_checkpoint()
            
            # save last snapshot
            if cfg.checkpoint.save_last_snapshot:
                self.workspace.save_snapshot()
            
            # topk checkpoint management
            if self.topk_manager is not None:
                # get metrics for topk evaluation
                metric_dict = {}
                logged_metrics = trainer.logged_metrics
                for key, value in logged_metrics.items():
                    if isinstance(value, torch.Tensor):
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value.item()
                
                # get topk checkpoint path
                topk_ckpt_path = self.topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    self.workspace.save_checkpoint(path=topk_ckpt_path)


class DiffusionPolicyMultiGpuLightningModule(pl.LightningModule):
    def __init__(self, cfg: OmegaConf, output_dir):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.output_dir = output_dir

        # Do not load the model here to avoid memory usage
        self.model = None
        self.ema_model = None
        self.ema = None
        
        # store training batch for sampling
        self.train_sampling_batch = None
        
        # environment runner for validation (only created on rank 0)
        self.env_runner = None
        
        # json logger (only on rank 0)
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
            # If the model is not yet configured, configure it here (without normalizer)
            if self.model is None:
                self.configure_model()
                
            # only create these on main process
            if self.trainer.is_global_zero:
                self.env_runner = hydra.utils.instantiate(
                    self.cfg.task.env_runner,
                    output_dir=self.output_dir
                )
                
                # setup json logger
                log_path = os.path.join(self.output_dir, 'logs.json.txt')
                # create the file if it does not exist
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
            train_dataloader_len = len(self.trainer.datamodule.train_dataloader())
        except Exception:
            raise ValueError("Please set `dataloader_len` in the config if the dataset does not support __len__.")
            
            
        num_training_steps = (
            train_dataloader_len * self.cfg.training.num_epochs
        ) // self.cfg.training.gradient_accumulate_every
        
        lr_scheduler = get_scheduler(
            self.cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # update every step
                "frequency": 1,
            }
        }
    
    def training_step(self, batch, batch_idx):
        # Ensure the model is already configured
        assert self.model is not None, "Model should be configured before training"
        
        # save first batch for sampling
        if self.train_sampling_batch is None:
            self.train_sampling_batch = {
                'obs': dict_apply(batch['obs'], lambda x: x.cpu()),
                'action': batch['action'].cpu()
            }
            
        # compute loss (following original code logic)
        raw_loss = self.model.compute_loss(batch)
        loss = raw_loss / self.cfg.training.gradient_accumulate_every
        
        # update EMA (only on main process)
        if self.cfg.training.use_ema and self.trainer.is_global_zero and self.ema is not None:
            self.ema.step(self.model)
        
        # log metrics (following original code structure)
        self.log('train_loss', raw_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, sync_dist=False)
        
        # log to json logger (only on main process, following original code)
        if (self.trainer.is_global_zero and 
        hasattr(self, 'json_logger') and 
        self.json_logger is not None):
            step_log = {
                'train_loss': raw_loss.item(),
                'global_step': self.global_step,
                'epoch': self.current_epoch,
                'lr': self.trainer.optimizers[0].param_groups[0]['lr']
            }
            # only log intermediate steps, final step is logged in on_validation_epoch_end
            if not self._is_last_batch_in_epoch(batch_idx):
                self.json_logger.log(step_log)
        
        return loss
    
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
        """Run environment tests and sampling at the end of validation epoch"""
        if not self.trainer.is_global_zero:
            return
            
        step_log = {}
        
        # select policy to evaluate
        policy = self.ema_model if (self.cfg.training.use_ema and self.ema_model is not None) else self.model
        assert policy is not None, "Policy should be available for evaluation"
        policy.eval()
        
        # run environment rollout (following original code frequency control)
        if (self.current_epoch % self.cfg.training.rollout_every) == 0:
            if self.env_runner is not None:
                runner_log = self.env_runner.run(policy)
                # log environment test results to both wandb and step_log
                step_log.update(runner_log)
                for key, value in runner_log.items():
                    self.log(f'env/{key}', value, on_epoch=True)
            else:
                print("Warning: env_runner is not set, skipping environment evaluation.")
        
        # run diffusion sampling (following original code frequency control)
        if (self.current_epoch % self.cfg.training.sample_every) == 0:
            if self.train_sampling_batch is not None:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    obs_dict = dict_apply(
                        self.train_sampling_batch['obs'], 
                        lambda x: x.to(self.device)
                    )
                    gt_action = self.train_sampling_batch['action'].to(self.device)
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    
                    step_log['train_action_mse_error'] = mse.item()
                    self.log('train_action_mse_error', mse, on_epoch=True)
                    
                    # clean up memory like in original code
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse
            else:
                print("Warning: train_sampling_batch is not set, skipping sampling evaluation.")
        
        # checkpoint handling is now handled at workspace level
        # keeping this section minimal to match original code structure
        
        # log final step metrics to json logger (following original code)
        if self.json_logger is not None:
            final_log = {
                'global_step': self.global_step,
                'epoch': self.current_epoch,
            }
            
            # add train_loss epoch average (equivalent to original code)
            if 'train_loss_epoch' in self.trainer.logged_metrics:
                final_log['train_loss'] = float(self.trainer.logged_metrics['train_loss_epoch'].item())
            
            # add all other logged metrics
            logged_metrics = self.trainer.logged_metrics
            for key, value in logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    final_log[key] = float(value.item())
            
            # add step_log metrics
            final_log.update(step_log)
            
            # log to json file
            self.json_logger.log(final_log)
        
        policy.train()
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # this ensures proper cleanup and follows original code structure
        pass
    
    def on_fit_end(self):
        """Called when training ends"""
        # close json logger properly
        if self.trainer.is_global_zero and self.json_logger is not None:
            self.json_logger.close()

class DiffusionMultiGpuDataModule(pl.LightningDataModule):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        # configure dataset
        self.dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        self.val_dataset = self.dataset.get_validation_dataset()
        
        # get normalizer
        self.normalizer = self.dataset.get_normalizer()
        
    def train_dataloader(self):
        return DataLoader(self.dataset, **self.cfg.dataloader)
    
    def val_dataloader(self):
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

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # create data module
        data_module = DiffusionMultiGpuDataModule(cfg)
        data_module.setup()
        
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
        
        # add original workspace checkpoint callback to replicate exact behavior
        checkpoint_callback = OriginalWorkspaceCheckpointCallback(self)
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
            check_val_every_n_epoch=cfg.training.val_every,
            precision=cfg.training.get('precision', 32),
            enable_progress_bar=True,
            enable_model_summary=True,
            # add debug mode support (following original code)
            fast_dev_run=cfg.training.get('debug', False),
            limit_train_batches=cfg.training.get('max_train_steps', None),
            limit_val_batches=cfg.training.get('max_val_steps', None),
            num_sanity_val_steps=0
        )
        
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
