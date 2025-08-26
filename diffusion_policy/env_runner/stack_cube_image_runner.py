"""
UR10e + Robotiq PickCube Image Runner
Ported from diffusion policy runner, adapted for ManiSkill
"""

import sys
import os
from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import time
import pathlib
import collections
import tqdm
import math
import wandb
import wandb.sdk.data_types.video as wv
from mani_skill.utils.wrappers import RecordEpisode

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from .stack_cube_env import StackCubeEnv
from .stack_cube_multicamera_env import StackCubeMultiCameraEnv

import mani_skill.envs
from .robotiq_ur10e.ur10e_robotiq import UR10eRobotiq

class StackCubeImageRunner(BaseImageRunner):
    """UR10e + Robotiq StackCube image runner using ManiSkill"""
    
    def __init__(self, 
                 output_dir,
                 env_id="StackCube-Customized", 
                 control_mode="pd_joint_pos", 
                 render_mode="rgb_array",
                 n_train=10,
                 n_train_vis=3,
                 train_start_seed=0,
                 n_test=22,
                 n_test_vis=6,
                 test_start_seed=10000,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 tqdm_interval_sec=5.0):
        """
        Initialize runner
        
        Args:
            output_dir: Output directory for videos and logs
            env_id: Environment ID
            control_mode: Control mode
            render_mode: Render mode
            n_train: Number of training episodes
            n_train_vis: Number of training episodes to record
            train_start_seed: Starting seed for training
            n_test: Number of test episodes
            n_test_vis: Number of test episodes to record
            test_start_seed: Starting seed for testing
            max_steps: Maximum steps per episode
            n_obs_steps: Number of observation steps
            n_action_steps: Number of action steps
            tqdm_interval_sec: Progress bar update interval
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.env_id = env_id
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        
        # Environment lists for policy evaluation
        self.env_seeds = []
        self.env_prefixs = []
        
        # Setup seeds and prefixes
        for i in range(n_train):
            seed = train_start_seed + i
            self.env_seeds.append(seed)
            self.env_prefixs.append('train/')
            
        for i in range(n_test):
            seed = test_start_seed + i
            self.env_seeds.append(seed)
            self.env_prefixs.append('test/')
        
        self.env = None
        self.agent = None
        
    def create_single_env(self, seed, record_video=True, video_filename=None):
        """
        Create a single environment instance
        
        Args:
            seed: Random seed
            record_video: Whether to record video
            video_filename: Custom video filename (for wandb compatibility)
            
        Returns:
            Configured environment
        """
        # Create environment
        env = gym.make(
            self.env_id,
            obs_mode="state_dict+rgb",
            robot_uids=UR10eRobotiq.uid,
            render_mode=self.render_mode,
            control_mode=self.control_mode,
            max_episode_steps=self.max_steps,
        )

        if record_video:
            video_dir = self.output_dir / "media"
            video_dir.mkdir(exist_ok=True)
            
            # RecordEpisode wrapper only needs output directory, will auto-generate filename
            env = RecordEpisode(
                env,
                str(video_dir),  # Only pass directory path
                max_steps_per_video=self.max_steps,
                save_video=True
            )
        
        return env
    
    def run(self, policy: BaseImagePolicy):
        """
        Run policy evaluation similar to pusht_image_runner
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            dict: Evaluation results and metrics
        """
        device = policy.device
        dtype = policy.dtype
        
        # Allocate data storage
        n_inits = len(self.env_seeds)
        all_video_paths: List[Optional[str]] = [None] * n_inits
        all_rewards = [0.0] * n_inits
        
        # Run evaluation for each seed
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            
            # Determine if we should record video
            if prefix == 'train/':
                episode_idx = i
                record_video = episode_idx < self.n_train_vis
            else:  # test
                episode_idx = i - self.n_train
                record_video = episode_idx < self.n_test_vis
            
            print(f"Running {prefix}episode {episode_idx + 1} (seed={seed})")
            
            # Create environment for this episode
            try:                
                env = self.create_single_env(seed=seed, record_video=record_video)
            except Exception as e:
                print(f"Error creating environment: {e}")
                import traceback
                traceback.print_exc()
                all_rewards[i] = 0.0
                all_video_paths[i] = None
                continue
            
            try:
                # Reset environment and policy
                obs, info = env.reset(seed=seed)
                policy.reset()
                
                # Convert ManiSkill obs to policy-expected format
                def process_obs(raw_obs):
                    """Convert ManiSkill observation to policy format"""
                    processed_obs = {}
                    
                    # Extract image from sensor_data
                    if 'sensor_data' in raw_obs and 'render_camera' in raw_obs['sensor_data']:
                        # ManiSkill returns RGB image, handle different dimensions
                        rgb_image = raw_obs['sensor_data']['render_camera']['rgb']
                        if isinstance(rgb_image, torch.Tensor):
                            # Handle different image dimensions
                            if len(rgb_image.shape) == 4:  # [B, H, W, C]
                                # Remove batch dimension and convert from [H, W, C] to [C, H, W]
                                rgb_image = rgb_image.squeeze(0).permute(2, 0, 1).float() / 255.0
                            elif len(rgb_image.shape) == 3:  # [H, W, C]
                                # Convert from [H, W, C] to [C, H, W]
                                rgb_image = rgb_image.permute(2, 0, 1).float() / 255.0
                            else:
                                print(f"Unexpected image shape: {rgb_image.shape}")
                                # continue
                            processed_obs['base_image'] = rgb_image.cpu().numpy()

                    if 'sensor_data' in raw_obs and 'mount_camera' in raw_obs['sensor_data']:
                        # ManiSkill returns RGB image, handle different dimensions
                        rgb_image = raw_obs['sensor_data']['mount_camera']['rgb']
                        if isinstance(rgb_image, torch.Tensor):
                            # Handle different image dimensions
                            if len(rgb_image.shape) == 4:  # [B, H, W, C]
                                # Remove batch dimension and convert from [H, W, C] to [C, H, W]
                                rgb_image = rgb_image.squeeze(0).permute(2, 0, 1).float() / 255.0
                            elif len(rgb_image.shape) == 3:  # [H, W, C]
                                # Convert from [H, W, C] to [C, H, W]
                                rgb_image = rgb_image.permute(2, 0, 1).float() / 255.0
                            else:
                                print(f"Unexpected image shape: {rgb_image.shape}")
                                # continue
                            processed_obs['wrist_image'] = rgb_image.cpu().numpy()
                        
                    # Extract agent_pos - 14D state vector
                    # Composition: joint_positions(6D) + ee_pos_quat(7D) + gripper_position(1D)
                    agent_pos_vector = []
                    
                    # 1. Joint positions (6D) - extract from first 6 elements of qpos
                    if 'agent' in raw_obs and 'qpos' in raw_obs['agent']:
                        qpos = raw_obs['agent']['qpos']
                        if isinstance(qpos, torch.Tensor):
                            joint_positions = qpos.flatten().cpu().numpy()[:6]
                            agent_pos_vector.extend(joint_positions)
                    
                    # 2. End-effector pose (7D) - TCP position and orientation
                    if 'extra' in raw_obs and 'tcp_pose' in raw_obs['extra']:
                        tcp_pose = raw_obs['extra']['tcp_pose']
                        if isinstance(tcp_pose, torch.Tensor):
                            ee_pos_quat = tcp_pose.flatten().cpu().numpy()
                            agent_pos_vector.extend(ee_pos_quat)
                    
                    # 3. Gripper position (1D) - extract 7th element from qpos
                    if 'agent' in raw_obs and 'qpos' in raw_obs['agent']:
                        qpos = raw_obs['agent']['qpos']
                        if isinstance(qpos, torch.Tensor):
                            gripper_position = qpos.flatten().cpu().numpy()[6:7]
                            agent_pos_vector.extend(gripper_position)
                    
                    # Ensure exactly 14 dimensions
                    if len(agent_pos_vector) == 14:
                        processed_obs['agent_pos'] = np.array(agent_pos_vector, dtype=np.float32)
                    else:
                        print(f"Warning: agent_pos dimension mismatch, expected 14, got {len(agent_pos_vector)}")
                        exit()
                    # If no specific mappings found, include all relevant state info
                    state_vector = []
                    
                    # Add agent joint positions
                    if 'agent' in raw_obs and 'qpos' in raw_obs['agent']:
                        qpos = raw_obs['agent']['qpos']
                        if isinstance(qpos, torch.Tensor):
                            state_vector.extend(qpos.flatten().cpu().numpy())
                    
                    # Add TCP pose
                    if 'extra' in raw_obs and 'tcp_pose' in raw_obs['extra']:
                        tcp_pose = raw_obs['extra']['tcp_pose']
                        if isinstance(tcp_pose, torch.Tensor):
                            state_vector.extend(tcp_pose.flatten().cpu().numpy())
                    
                    # Add object and goal positions
                    if 'extra' in raw_obs:
                        for key in ['obj_pose', 'goal_pos', 'tcp_to_obj_pos', 'obj_to_goal_pos']:
                            if key in raw_obs['extra']:
                                val = raw_obs['extra'][key]
                                if isinstance(val, torch.Tensor):
                                    state_vector.extend(val.flatten().cpu().numpy())
                    
                    if state_vector:
                        processed_obs['state'] = np.array(state_vector)
                    
                    return processed_obs
                
                obs = process_obs(obs)
                
                # Get observation shape and create observation history
                obs_deque = collections.deque(maxlen=self.n_obs_steps)
                
                # Fill initial observation history
                for _ in range(self.n_obs_steps):
                    obs_deque.append(obs)
                
                total_reward = 0.0
                
                # Run episode with policy
                pbar = tqdm.tqdm(total=self.max_steps, 
                               desc=f"Eval PickCubeImageRunner {i+1}/{n_inits}", 
                               leave=False, mininterval=self.tqdm_interval_sec)
                
                done = False
                step_count = 0
                
                while not done and step_count < self.max_steps:
                    try:
                        # Create observation dictionary
                        obs_dict = {}
                        
                        # Stack observations if we have multiple timesteps
                        if len(obs_deque) > 0:
                            stacked_obs = {}
                            for key in obs_deque[0].keys():
                                # Stack observations along time dimension
                                obs_list = [obs_data[key] for obs_data in obs_deque]
                                if isinstance(obs_list[0], np.ndarray):
                                    stacked_obs[key] = np.stack(obs_list, axis=0)
                                else:
                                    stacked_obs[key] = obs_list[-1]  # Use latest if not stackable
                            obs_dict.update(stacked_obs)
                        
                        # Convert to tensors
                        tensor_obs_dict = {}
                        
                        for key, value in obs_dict.items():
                            # Skip 'state' key as it's not expected by the policy normalizer
                            if key == 'state':
                                continue
                                
                            try:
                                if isinstance(value, torch.Tensor):
                                    # For torch tensors, ensure correct device/dtype and add batch dimension
                                    tensor_value = value.to(device=device, dtype=dtype)
                                    if len(tensor_value.shape) == 0:  # scalar tensor
                                        tensor_obs_dict[key] = tensor_value.unsqueeze(0)
                                    else:
                                        tensor_obs_dict[key] = tensor_value.unsqueeze(0)
                                        
                                elif isinstance(value, np.ndarray):
                                    # For numpy arrays, convert to torch and add batch dimension
                                    tensor_obs_dict[key] = torch.from_numpy(value).to(device=device, dtype=dtype).unsqueeze(0)
                                    
                                elif isinstance(value, (int, float)):
                                    # For scalar values
                                    tensor_obs_dict[key] = torch.tensor([value], device=device, dtype=dtype)
                                    
                                else:
                                    # Skip unsupported types
                                    continue
                                    
                            except Exception as e:
                                print(f"Error converting '{key}' to tensor: {e}")
                                continue
                        
                        # Run policy to get action sequence
                        with torch.no_grad():
                            action_dict = policy.predict_action(tensor_obs_dict)
                        
                        # Extract action sequence (similar to MultiStepWrapper.step)
                        if isinstance(action_dict['action'], torch.Tensor):
                            action_sequence = action_dict['action'].detach().to('cpu').numpy()
                            # Handle different shapes: [B, n_action_steps, action_dim] -> [n_action_steps, action_dim]
                            if len(action_sequence.shape) == 3:  # [B=1, n_action_steps, action_dim]
                                action_sequence = action_sequence[0]  # [n_action_steps, action_dim]
                            elif len(action_sequence.shape) == 2:  # [n_action_steps, action_dim]
                                pass  # Already correct shape
                            else:
                                # Single action, expand to sequence
                                action_sequence = action_sequence.reshape(1, -1)
                        else:
                            action_sequence = np.array([action_dict['action']])
                        
                        # Execute action sequence (similar to MultiStepWrapper.step loop)
                        episode_rewards = []
                        episode_dones = []
                        
                        for action_idx in range(min(self.n_action_steps, len(action_sequence))):
                            if done:
                                break
                                
                            action = action_sequence[action_idx]
                            
                            try:
                                obs, reward, terminated, truncated, info = env.step(action)
                                done = terminated or truncated
                                episode_rewards.append(float(reward))
                                episode_dones.append(done)
                                
                                # Process new observation and update history
                                processed_obs = process_obs(obs)
                                obs_deque.append(processed_obs)
                                
                                step_count += 1
                                pbar.update(1)
                                
                                if done:
                                    break
                                    
                            except Exception as e:
                                print(f"Error executing action {action_idx} at step {step_count}: {e}")
                                import traceback
                                traceback.print_exc()
                                done = True
                                break
                        
                        # Aggregate rewards (similar to MultiStepWrapper reward aggregation)
                        if episode_rewards:
                            step_reward = np.max(episode_rewards)  # Use 'max' aggregation like MultiStepWrapper
                            total_reward += step_reward
                    
                    except Exception as e:
                        print(f"Error in episode step {step_count}: {e}")
                        break
                
                pbar.close()
                
                # Store results
                all_rewards[i] = total_reward
                
                # Immediately upload video to wandb after episode completion
                if record_video:
                    try:
                        video_dir = self.output_dir / "media"
                        # Find the most recently generated video file
                        video_files = list(video_dir.glob("*.mp4"))
                        if video_files:
                            # Sort by modification time and get the latest one
                            latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                            video_path = str(latest_video)
                            print(f"Found video file: {latest_video}")
                            
                            # Immediately upload to wandb to prevent overwriting
                            try:
                                video_file = pathlib.Path(video_path)
                                if video_file.exists() and video_file.stat().st_size > 0:
                                    print(f"Video found: {video_path} (size: {video_file.stat().st_size} bytes)")
                                    
                                    # Check if wandb is initialized before uploading
                                    if wandb.run is not None:
                                        print(f"Uploading video to wandb: {video_path}")
                                        sim_video = wandb.Video(str(video_file))
                                        
                                        # Log immediately with episode-specific key
                                        episode_log_data = {
                                            prefix + f'sim_video_{seed}': sim_video,
                                            prefix + f'sim_max_reward_{seed}': total_reward
                                        }
                                        wandb.log(episode_log_data)
                                        print(f"Video and reward logged to wandb: {prefix}sim_video_{seed}")
                                    else:
                                        print("wandb not initialized, skipping video upload")
                                    
                                    all_video_paths[i] = video_path  # Store for reference
                                else:
                                    print(f"Video file is empty or doesn't exist: {video_path}")
                                    all_video_paths[i] = None
                            except Exception as e:
                                print(f"Error processing video: {e}")
                                import traceback
                                traceback.print_exc()
                                all_video_paths[i] = video_path  # Store path even if upload failed
                        else:
                            print(f"No video files found in {video_dir}")
                            all_video_paths[i] = None
                    except Exception as e:
                        print(f"Error processing video: {e}")
                        all_video_paths[i] = None
                else:
                    all_video_paths[i] = None
                
                print(f"Episode completed: reward={total_reward:.3f}, steps={step_count}")
                
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                all_rewards[i] = 0.0
                all_video_paths[i] = None
            finally:
                try:
                    env.close()
                except Exception as e:
                    print(f"Error closing environment: {e}")
        
        # Compile results
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = all_rewards[i] if all_rewards[i] is not None else 0.0
            max_rewards[prefix].append(max_reward)
            
            # Note: Individual episode rewards and videos were already logged immediately
            # We only log the reward here for completeness in case wandb logging failed earlier
            if prefix + f'sim_max_reward_{seed}' not in log_data:
                log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
        
        # Log aggregate metrics
        for prefix, rewards in max_rewards.items():
            name = prefix + 'mean_score'
            mean_score = np.mean(rewards)
            log_data[name] = mean_score
            print(f"{prefix}mean_score: {mean_score:.3f}")
        
        return log_data
    
    def test_random_actions(self, n_episodes=5, debug=False):
        """
        Test function to run random actions (for debugging and environment testing)
        
        Args:
            n_episodes: Number of episodes to run
            debug: Whether to enable debug output
            
        Returns:
            dict: Test results
        """
        print("=== Testing Environment with Random Actions ===")
        
        results = []
        
        for i in range(n_episodes):
            seed = self.train_start_seed + i
            print(f"Running random test episode {i+1}/{n_episodes} (seed={seed})")
            
            # Create environment
            try:
                env = self.create_single_env(seed=seed, record_video=(i < 2), video_filename=f"test_{seed}.mp4")
            except Exception as e:
                print(f"Error creating environment for random test: {e}")
                results.append({
                    'seed': seed,
                    'steps': 0,
                    'total_reward': 0.0,
                    'success': False,
                    'error': f"Environment creation failed: {str(e)}"
                })
                continue
            
            try:
                # Reset environment
                obs, info = env.reset(seed=seed)
                
                total_reward = 0.0
                steps_executed = 0
                
                for step in range(self.max_steps):
                    steps_executed = step + 1
                    
                    try:
                        # Sample random action
                        action = env.action_space.sample()
                        
                        # Execute action
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += float(reward)
                        
                        if terminated or truncated:
                            if debug:
                                print(f"Episode terminated at step {step}")
                            break
                            
                    except Exception as e:
                        print(f"Error in random action step {step}: {e}")
                        break
                
                result = {
                    'seed': seed,
                    'steps': steps_executed,
                    'total_reward': total_reward,
                    'success': total_reward > 0.1  # Simple success criterion
                }
                
                results.append(result)
                
                if debug:
                    print(f"Episode {i+1}: {steps_executed} steps, reward: {total_reward:.3f}")
                else:
                    status = "✅" if result['success'] else "❌"
                    print(f"Test {i+1}: {status} (reward: {total_reward:.3f})")
                    
            except Exception as e:
                print(f"Error in random test episode {i+1}: {e}")
                results.append({
                    'seed': seed,
                    'steps': 0,
                    'total_reward': 0.0,
                    'success': False,
                    'error': str(e)
                })
            finally:
                try:
                    env.close()
                except Exception as e:
                    print(f"Error closing random test environment: {e}")
        
        # Summary
        success_rate = np.mean([r['success'] for r in results])
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        print(f"\n=== Random Action Test Summary ===")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Average steps: {avg_steps:.1f}")
        
        return {
            'results': results,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        }


def main():
    """Main function - demonstration usage"""
    print("=== UR10e + Robotiq PickCube Image Runner ===")
    
    # Create runner
    output_dir = "./evaluation_results"
    
    try:
        runner = PickCubeImageRunner(
            output_dir=output_dir,
            n_train=10,
            n_train_vis=3,
            n_test=22,
            n_test_vis=6
        )
        
        # Test environment creation first
        print("\nTesting environment creation...")
        temp_env = runner.create_single_env(seed=42, record_video=False)
        print(f"Action space: {temp_env.action_space}")
        
        # Test environment reset
        obs, info = temp_env.reset(seed=42)
        print(f"Observation type: {type(obs)}")
        
        # Test a single step
        action = temp_env.action_space.sample()
        obs, reward, terminated, truncated, info = temp_env.step(action)
        print("Environment step successful")
        
        temp_env.close()
        
        # Test with random actions
        print("\nTesting random actions...")
        test_results = runner.test_random_actions(n_episodes=3, debug=True)
        
        # Test RandomPolicy creation
        print("\nTesting RandomPolicy creation...")
        temp_env = runner.create_single_env(seed=42, record_video=False)
        
        random_policy = RandomPolicy(
            action_space=temp_env.action_space,
            shape_meta={'action': {'shape': temp_env.action_space.shape}},
            device='cpu',
            dtype=torch.float32
        )
        
        temp_env.close()
        
        # Test policy interface with dummy observations
        print("\nTesting policy interface...")
        batch_size = 1
        dummy_obs = {
            'image': torch.randn(batch_size, 8, 3, 96, 96, device=random_policy.device, dtype=random_policy.dtype),
            'agent_pos': torch.randn(batch_size, 8, 2, device=random_policy.device, dtype=random_policy.dtype)
        }
        
        with torch.no_grad():
            action_dict = random_policy.predict_action(dummy_obs)
        print(f"Action shape: {action_dict['action'].shape}")
        
        # Test policy evaluation
        print("\nTesting policy evaluation...")
        
        # Create smaller test configuration
        runner.n_train = 2
        runner.n_test = 2
        runner.env_seeds = []
        runner.env_prefixs = []
        
        # Setup seeds and prefixes for reduced episodes
        for i in range(2):
            seed = runner.train_start_seed + i
            runner.env_seeds.append(seed)
            runner.env_prefixs.append('train/')
            
        for i in range(2):
            seed = runner.test_start_seed + i
            runner.env_seeds.append(seed)
            runner.env_prefixs.append('test/')
        
        results = runner.run(random_policy)
        
        print(f"\n=== Final Results ===")
        print(f"Output directory: {output_dir}")
        print(f"Training mean score: {results.get('train/mean_score', 0.0):.3f}")
        print(f"Testing mean score: {results.get('test/mean_score', 0.0):.3f}")
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
