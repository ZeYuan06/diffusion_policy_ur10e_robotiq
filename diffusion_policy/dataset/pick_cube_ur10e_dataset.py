from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PickCubeUR10EDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
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
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        image = sample['img'].astype(np.float32) / 255.0

        data = {
            'obs': {
                'image': image, # T, 3, 180, 320
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


def test():
    import os
    
    # 使用 gello_zarr 数据集进行测试
    zarr_path = "/home/zy3722/diffusion_policy_ur10e_robotiq/data/sample.zarr"
    
    print(f"Testing with zarr path: {zarr_path}")
    print(f"Path exists: {os.path.exists(zarr_path)}")
    
    try:
        # 创建数据集，设置小的horizon进行快速测试
        dataset = PickCubeUR10EDataset(
            zarr_path, 
            horizon=8,
            val_ratio=0.2,
            pad_before=1,
            pad_after=1
        )
        
        print(f"Dataset created successfully!")
        print(f"Total episodes: {dataset.replay_buffer.n_episodes}")
        print(f"Total steps: {dataset.replay_buffer.n_steps}")
        print(f"Dataset length (samples): {len(dataset)}")
        
        # 检查数据形状
        print("\nData shapes:")
        for key in ['img', 'state', 'action']:
            if key in dataset.replay_buffer.root['data']:
                shape = dataset.replay_buffer[key].shape
                print(f"  {key}: {shape}")
        
        # 测试获取一个样本
        print("\nTesting sample retrieval...")
        sample = dataset[0]
        print("Sample keys:", list(sample.keys()))
        
        # 简化的数据访问测试
        print("\nSample data info:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'shape'):
                        print(f"    {sub_key}: shape={sub_value.shape}, dtype={sub_value.dtype}")
        
        # 详细检查 agent pos 和 action 信息
        print("\n=== Agent Position and Action Analysis ===")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            
            # Agent position 信息
            if 'obs' in sample and hasattr(sample['obs'], '__getitem__'):
                try:
                    agent_pos = sample['obs']['agent_pos']
                    print(f"  Agent pos shape: {agent_pos.shape}")
                    print(f"  Agent pos (first 3 timesteps):")
                    for t in range(min(3, agent_pos.shape[0])):
                        pos = agent_pos[t].numpy() if hasattr(agent_pos[t], 'numpy') else agent_pos[t]
                        print(f"    t={t}: [{pos[0]:.4f}, {pos[1]:.4f}]")
                except Exception as e:
                    print(f"  Error accessing agent_pos: {e}")
            
            # Action 信息
            if 'action' in sample:
                try:
                    action = sample['action']
                    print(f"  Action shape: {action.shape}")
                    print(f"  Action (first 3 timesteps):")
                    for t in range(min(3, action.shape[0])):
                        act = action[t].numpy() if hasattr(action[t], 'numpy') else action[t]
                        print(f"    t={t}: [{act[0]:.4f}, {act[1]:.4f}]")
                    
                    # 计算动作统计信息
                    action_np = action.numpy() if hasattr(action, 'numpy') else action
                    print(f"  Action statistics:")
                    print(f"    Mean: [{action_np.mean(axis=0)[0]:.4f}, {action_np.mean(axis=0)[1]:.4f}]")
                    print(f"    Std:  [{action_np.std(axis=0)[0]:.4f}, {action_np.std(axis=0)[1]:.4f}]")
                    print(f"    Min:  [{action_np.min(axis=0)[0]:.4f}, {action_np.min(axis=0)[1]:.4f}]")
                    print(f"    Max:  [{action_np.max(axis=0)[0]:.4f}, {action_np.max(axis=0)[1]:.4f}]")
                except Exception as e:
                    print(f"  Error accessing action: {e}")
        
        # 保存图像
        print("\n=== Saving Sample Images ===")
        try:
            import matplotlib.pyplot as plt
            import os
            
            # 创建保存目录
            save_dir = "/home/zy3722/diffusion_policy_ur10e_robotiq/test_images"
            os.makedirs(save_dir, exist_ok=True)
            
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                if 'obs' in sample and hasattr(sample['obs'], '__getitem__'):
                    try:
                        image = sample['obs']['image']  # shape: (T, 3, H, W)
                        image_np = image.numpy() if hasattr(image, 'numpy') else image
                        
                        # 随机抽样时间步保存图像
                        num_timesteps = image_np.shape[0]
                        num_samples = min(3, num_timesteps)
                        
                        # 随机选择时间步，确保不重复
                        import random
                        random_timesteps = sorted(random.sample(range(num_timesteps), num_samples))
                        print(f"Randomly selected timesteps: {random_timesteps} from total {num_timesteps}")
                        
                        for idx, t in enumerate(random_timesteps):
                            # Convert from (3, H, W) to (H, W, 3)
                            # fixme: 这里原始形状是(320, 3, 180)
                            print(f"Shape of image at timestep {t}: {image_np[t].shape}")
                            img_t = np.transpose(image_np[t], (2, 0, 1))
                            
                            # 确保图像值在 [0, 1] 范围内
                            img_t = np.clip(img_t, 0, 1)
                            
                            # 保存图像
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img_t)
                            plt.title(f'Sample {i}, Timestep {t} (random #{idx+1})')
                            plt.axis('off')
                            
                            filename = f"sample_{i}_timestep_{t}_random_{idx+1}.png"
                            filepath = os.path.join(save_dir, filename)
                            plt.savefig(filepath, bbox_inches='tight', dpi=100)
                            plt.close()
                            
                            print(f"  Saved: {filepath}")
                    except Exception as e:
                        print(f"  Error saving images for sample {i}: {e}")
            
            print(f"Images saved to: {save_dir}")
            
        except ImportError:
            print("Matplotlib not available, skipping image saving")
        except Exception as e:
            print(f"Error during image saving: {e}")
        
        # 测试验证集
        print("\nTesting validation dataset...")
        val_dataset = dataset.get_validation_dataset()
        print(f"Validation dataset length: {len(val_dataset)}")
        
        # 测试normalizer
        print("\nTesting normalizer...")
        normalizer = dataset.get_normalizer()
        print(f"Normalizer created successfully")
        
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()
