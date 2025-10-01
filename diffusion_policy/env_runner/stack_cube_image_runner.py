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
from .pick_box_env import PickBoxEnv

import mani_skill.envs
from .robotiq_ur10e.ur10e_robotiq import UR10eRobotiq

class StackCubeImageRunner(BaseImageRunner):
    """UR10e + Robotiq StackCube image runner using ManiSkill"""
    
    def __init__(self, 
                 output_dir,
                 env_id="StackCube-Customized", 
                 control_mode="pd_joint_delta_pos", 
                 render_mode="rgb_array",
                 n_test=5,
                 n_test_vis=5,
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
        for i in range(n_test):
            seed = test_start_seed + i
            self.env_seeds.append(seed)
            self.env_prefixs.append('test/')
        
        self.env = None
        self.agent = None
        
    def create_single_env(self, record_video=True, video_filename=None):
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

    def run(self, policy: BaseImagePolicy, current_step=None) -> Dict:
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
        all_rewards = [0.0] * n_inits
        
        # Run evaluation for each seed
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            
            # Determine if we should record video
            episode_idx = i
            record_video = episode_idx < self.n_test_vis
            
            print(f"Running {prefix}episode {episode_idx + 1} (seed={seed})")
            
            # Create environment for this episode
            try:                
                env = self.create_single_env(record_video=record_video)
            except Exception as e:
                print(f"Error creating environment: {e}")
                import traceback
                traceback.print_exc()
                all_rewards[i] = 0.0
                continue
            
            try:
                # Reset environment and policy
                obs, info = env.reset(seed=seed)
                policy.reset()
                
                # Convert ManiSkill obs to policy-expected format
                def process_obs(raw_obs, env):
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
                            # processed_obs['image'] = rgb_image.cpu().numpy()

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

                    env = env.unwrapped
                    robot = env.agent.robot
                    joint_positions = robot.get_qpos().cpu().numpy()[0, :6]

                    tcp_pose = env.agent.tcp.pose
                    tcp_pos = tcp_pose.p.cpu().numpy()[0]
                    tcp_quat = tcp_pose.q.cpu().numpy()[0]
                    ee_pos_quat = np.concatenate([tcp_pos, tcp_quat])

                    gripper_pos = env.unwrapped.agent.gripper_qpos
                    gripper_position = np.array([gripper_pos])

                    processed_obs['agent_pos'] = np.concatenate([joint_positions, ee_pos_quat, gripper_position], axis=0).astype(np.float32)

                    return processed_obs

                obs = process_obs(obs, env)

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

                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                            episode_rewards.append(float(reward))
                            episode_dones.append(done)
                            
                            # Process new observation and update history
                            processed_obs = process_obs(obs, env)
                            obs_deque.append(processed_obs)
                            
                            step_count += 1
                            pbar.update(1)
                            
                            if done:
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
                    video_dir = self.output_dir / "media"
                    # Find the most recently generated video file
                    video_files = list(video_dir.glob("*.mp4"))
                    if video_files:
                        # Sort by modification time and get the latest one
                        latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                        video_path = str(latest_video)
                        print(f"Found video file: {latest_video}")
                        
                        # Immediately upload to wandb to prevent overwriting
                        video_file = pathlib.Path(video_path)
                        if video_file.exists() and video_file.stat().st_size > 0:
                            print(f"Video found: {video_path} (size: {video_file.stat().st_size} bytes)")
                            
                            # Check if wandb is initialized before uploading
                            if wandb.run is not None:
                                print(f"Uploading video to wandb: {video_path}")
                                sim_video = wandb.Video(str(video_file))
                                wandb.log({prefix + f'sim_video_{seed}': sim_video}, step=current_step)
                                print(f"Video and reward logged to wandb: {prefix}sim_video_{seed}")
                            else:
                                print("wandb not initialized, skipping video upload")
                        else:
                            print(f"Video file is empty or doesn't exist: {video_path}")
                    else:
                        print(f"No video files found in {video_dir}")
                
                print(f"Episode completed: reward={total_reward:.3f}, steps={step_count}")
                
            except Exception as e:
                print(f"Error in episode {i}: {e}")
                all_rewards[i] = 0.0
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
            seed = self.test_start_seed + i
            print(f"Running random test episode {i+1}/{n_episodes} (seed={seed})")
            
            # Create environment
            try:
                env = self.create_single_env(record_video=(i < 2), video_filename=f"test_{seed}.mp4")
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
