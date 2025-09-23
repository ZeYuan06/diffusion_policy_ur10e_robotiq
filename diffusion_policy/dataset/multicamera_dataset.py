from typing import Dict
import torch
import numpy as np
import copy
import pickle
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class StackCubeUR10EDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            val_mask_save_path=None,
            load_existing_val_mask=True
            ):
        
        super().__init__()
        # Use lazy loading - create_from_path opens zarr file directly without loading to memory
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['base_img', 'wrist_img', 'state', 'action'])
        # self.replay_buffer = ReplayBuffer.create_from_path(zarr_path)
        
        # Verify that all required keys exist in the zarr file
        # required_keys = ['base_img', 'wrist_img', 'state', 'action']
        # available_keys = list(self.replay_buffer.keys())
        # missing_keys = set(required_keys) - set(available_keys)
        # if missing_keys:
        #     raise KeyError(f"Missing required keys in zarr file: {missing_keys}. "
        #                   f"Available keys: {available_keys}")

        self.val_mask_save_path = val_mask_save_path
        
        # Get or create validation mask with persistence
        val_mask = self._get_or_create_val_mask(val_ratio, seed, load_existing_val_mask)
        
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _get_or_create_val_mask(self, val_ratio, seed, load_existing=True):
        """Get or create validation mask with persistence to ensure train/val separation"""
        if self.val_mask_save_path and load_existing and os.path.exists(self.val_mask_save_path):
            # Load existing validation mask
            print(f"Loading existing validation mask from {self.val_mask_save_path}")
            with open(self.val_mask_save_path, 'rb') as f:
                mask_data = pickle.load(f)
                val_mask = mask_data['val_mask']
                saved_n_episodes = mask_data['n_episodes']
                
            # Verify dataset consistency
            current_n_episodes = self.replay_buffer.n_episodes
            if saved_n_episodes != current_n_episodes:
                print(f"Warning: Saved mask has {saved_n_episodes} episodes, "
                      f"but current dataset has {current_n_episodes} episodes")
                print("Regenerating validation mask...")
                val_mask = get_val_mask(
                    n_episodes=current_n_episodes, 
                    val_ratio=val_ratio,
                    seed=seed)
                self._save_val_mask(val_mask)
            else:
                print(f"Loaded validation mask: {val_mask.sum()} validation episodes, "
                      f"{(~val_mask).sum()} training episodes")
        else:
            # Create new validation mask
            print(f"Creating new validation mask with val_ratio={val_ratio}")
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes, 
                val_ratio=val_ratio,
                seed=seed)
            
            # Save validation mask
            if self.val_mask_save_path:
                self._save_val_mask(val_mask)
                
        return val_mask
    
    def _save_val_mask(self, val_mask):
        """Save validation mask to file for consistent train/val splits"""
        if self.val_mask_save_path:
            os.makedirs(os.path.dirname(self.val_mask_save_path), exist_ok=True)
            mask_data = {
                'val_mask': val_mask,
                'n_episodes': self.replay_buffer.n_episodes,
                'val_episodes': val_mask.sum(),
                'train_episodes': (~val_mask).sum(),
                'creation_time': np.datetime64('now')
            }
            with open(self.val_mask_save_path, 'wb') as f:
                pickle.dump(mask_data, f)
            print(f"Saved validation mask to {self.val_mask_save_path}")
            print(f"  Validation episodes: {val_mask.sum()}")
            print(f"  Training episodes: {(~val_mask).sum()}")

    def get_validation_dataset(self):
        """Get validation dataset using the saved validation mask"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask  # Use saved validation mask
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_train_only_dataset(self):
        """Get dataset with only training data (excludes validation episodes)"""
        train_set = copy.copy(self)
        train_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.train_mask  # Use training mask only
        )
        return train_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['base_image'] = get_image_range_normalizer()
        normalizer['wrist_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        base_image = sample['base_img'].astype(np.float32) / 255.0
        wrist_image = sample['wrist_img'].astype(np.float32) / 255.0

        data = {
            'obs': {
                'base_image': base_image, # T, 3, 512, 512
                'wrist_image': wrist_image, # T, 3, 512, 512
                'agent_pos': agent_pos, # T, 14
            },
            'action': sample['action'].astype(np.float32) # T, 7
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
