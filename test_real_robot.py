"""
Usage:
(robodiff)$ python test_real_robot.py -o <save_dir> --robot_ip <ip_of_ur5> --zarr_path <dataset_path>

================ Dataset Replay Test ==============
This script loads a dataset and replays the actions on a real robot.
The robot will execute the recorded trajectories from the dataset.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start dataset replay execution.
Press "Q" to exit program.
Press "S" to stop current episode and move to next.

Make sure you can hit the robot hardware emergency-stop button quickly! 
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.dataset.pick_cube_ur10e_dataset import PickCubeUR10EDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer


class EpisodeReplayPolicy(BaseImagePolicy):
    """
    Episode replay policy - replays complete episode data from dataset
    """
    
    def __init__(self, episodes, device='cpu', dtype=torch.float32):
        """
        Initialize Episode replay policy
        
        Args:
            episodes: Complete episode data list
            device: Device
            dtype: Data type
        """
        super().__init__()
        self.episodes = episodes
        self.current_episode_idx = 0
        self.current_step_idx = 0
        
        # Create dummy parameter to establish device and data type
        if isinstance(device, str):
            device = torch.device(device)
        dummy_param = torch.zeros(1, device=device, dtype=dtype, requires_grad=False)
        self.register_parameter('_dummy_param', torch.nn.parameter.Parameter(dummy_param))
    
    def predict_action(self, obs_dict):
        """
        Predict action - get next action from current episode
        
        Args:
            obs_dict: Observation dictionary, shape B,To,*
            
        Returns:
            Dictionary containing action tensor, shape B,Ta,Da
        """
        # Get batch size
        batch_size = list(obs_dict.values())[0].shape[0]
        
        # Get current episode
        if self.current_episode_idx >= len(self.episodes):
            print("All episodes replayed, restarting from beginning")
            self.current_episode_idx = 0
            self.current_step_idx = 0
        
        current_episode = self.episodes[self.current_episode_idx]
        episode_length = len(current_episode['action'])
        
        # Get current step action
        if self.current_step_idx < episode_length:
            action = current_episode['action'][self.current_step_idx]
            print(f"Episode {self.current_episode_idx}, Step {self.current_step_idx}/{episode_length}")
        else:
            # Current episode ended, move to next episode
            self.current_episode_idx += 1
            self.current_step_idx = 0
            
            if self.current_episode_idx < len(self.episodes):
                current_episode = self.episodes[self.current_episode_idx]
                action = current_episode['action'][0]
                print(f"Switching to Episode {self.current_episode_idx}")
            else:
                # All episodes finished, use zero action
                action = np.zeros(7)
                print("All episodes replayed, using zero action")
        
        # Increment step index
        self.current_step_idx += 1
        
        # Convert to tensor
        action_tensor = torch.tensor(
            action, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Expand to batch dimension and action sequence dimension
        action_tensor = action_tensor.unsqueeze(0).repeat(batch_size, 1)  # B, Da
        action_tensor = action_tensor.unsqueeze(1)  # B, Ta=1, Da
            
        return {'action': action_tensor}
    
    def reset(self):
        """Reset to beginning of first episode"""
        self.current_episode_idx = 0
        self.current_step_idx = 0
        print("Policy reset to Episode 0")
    
    def set_normalizer(self, normalizer):
        """Set normalizer"""
        self.normalizer = normalizer
        
    def forward(self, *args, **kwargs):
        """Forward pass"""
        pass
    
    def get_current_episode_info(self):
        """Get current episode information"""
        if self.current_episode_idx < len(self.episodes):
            episode_length = len(self.episodes[self.current_episode_idx]['action'])
            return {
                'episode_idx': self.current_episode_idx,
                'step_idx': self.current_step_idx,
                'episode_length': episode_length,
                'progress': f"{self.current_step_idx}/{episode_length}"
            }
        return None


def load_dataset_episodes(zarr_path, num_episodes=3):
    """
    Load episodes from dataset
    
    Args:
        zarr_path: Path to zarr dataset
        num_episodes: Number of episodes to load
        
    Returns:
        episodes: List of episode data
    """
    print(f"Loading dataset from: {zarr_path}")
    
    try:
        # Load replay buffer directly
        replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        
        print(f"Dataset loaded successfully!")
        print(f"Total episodes: {replay_buffer.n_episodes}")
        print(f"Total steps: {replay_buffer.n_steps}")
        
        # Extract episodes
        episodes = []
        episode_ends = replay_buffer.episode_ends
        
        # Select episodes to load
        available_episodes = list(range(len(episode_ends)))
        selected_episodes = available_episodes[:min(num_episodes, len(episode_ends))]
        
        for ep_idx in selected_episodes:
            # Calculate episode start and end indices
            start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            end_idx = int(episode_ends[ep_idx])
            episode_length = end_idx - start_idx
            
            print(f"Loading Episode {ep_idx}: {episode_length} steps (indices {start_idx}-{end_idx})")
            
            # Extract episode data
            episode_data = {
                'action': [],
                'state': [],
                'episode_idx': ep_idx,
                'episode_length': episode_length
            }
            
            # Extract data step by step
            for step_idx in range(start_idx, end_idx):
                # Action data
                action = np.array(replay_buffer['action'][step_idx]).astype(np.float32)
                episode_data['action'].append(action)
                
                # State data
                state = np.array(replay_buffer['state'][step_idx]).astype(np.float32)
                episode_data['state'].append(state)
            
            # Convert to numpy arrays
            episode_data['action'] = np.array(episode_data['action'])
            episode_data['state'] = np.array(episode_data['state'])
            
            print(f"  Action shape: {episode_data['action'].shape}")
            print(f"  State shape: {episode_data['state'].shape}")
            
            # Analyze action ranges
            actions = episode_data['action']
            print(f"  Action statistics:")
            print(f"    Range: [{actions.min():.3f}, {actions.max():.3f}]")
            if actions.shape[-1] >= 7:
                print(f"    Pose actions (0:6): [{actions[:, :6].min():.3f}, {actions[:, :6].max():.3f}]")
                print(f"    Gripper actions (6): [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")
            
            episodes.append(episode_data)
        
        print(f"Successfully loaded {len(episodes)} episodes")
        return episodes
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--zarr_path', '-z', required=True, help='Path to zarr dataset')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--num_episodes', '-ne', default=3, type=int, help='Number of episodes to replay from dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
def main(zarr_path, output, robot_ip, num_episodes,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
    # Check if dataset exists
    if not pathlib.Path(zarr_path).exists():
        print(f"Dataset path does not exist: {zarr_path}")
        return
    
    # Load episodes from dataset
    print(f"Loading episodes from dataset: {zarr_path}")
    episodes = load_dataset_episodes(zarr_path, num_episodes=num_episodes)
    if len(episodes) == 0:
        print("Failed to load any episodes from dataset")
        return
    
    # Create dataset replay policy
    policy = EpisodeReplayPolicy(episodes, device='cpu', dtype=torch.float32)
    print(f"Created replay policy with {len(episodes)} episodes")
    
    # Setup experiment parameters
    dt = 1/frequency
    action_offset = 0  # No action offset for dataset replay
    delta_action = False  # Dataset replay uses absolute actions
    device = torch.device('cpu')  # Use CPU for simple replay
    
    # Create default shape meta for observations (since we don't have a trained model config)
    shape_meta = {
        'obs': {
            'agent_pos': {'type': 'low_dim', 'shape': (14,)},
            'image': {'type': 'rgb', 'shape': (3, 180, 320)}
        },
        'action': {'type': 'low_dim', 'shape': (7,)},
        'n_obs_steps': 2
    }
    
    obs_res = (320, 180)  # Default camera resolution
    n_obs_steps = 2
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        # Remove SpaceMouse dependency and use RealEnv directly
        with RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            use_gripper=True,
            gripper_port=63352,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)

            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                print(f"Warmup action shape: {action.shape}")
                # Action should be 7D (6 pose + 1 gripper)
                assert action.shape[-1] >= 7, f"Expected action dimension >= 7, got {action.shape[-1]}"
                print("Dataset replay policy ready - actions include gripper control")
                del result

            print('Ready!')
            while True:
                # ========= automatic robot reset ==========
                print("Automatic robot reset to home position...")
                
                # Define home joint position from RealEnv: [0,-90,-90,-90,90,0] degrees
                # Converting to radians: [0, -pi/2, -pi/2, -pi/2, pi/2, 0]
                home_joints_deg = np.array([0, -90, -90, -90, 90, 0])
                home_joints = home_joints_deg / 180 * np.pi
                print(f"Home joints (rad): {home_joints}")
                print(f"Home joints (deg): {home_joints_deg}")
                
                # Get current robot state
                state = env.get_robot_state()
                current_joints = state.get('ActualQ', np.zeros(6))
                current_pose = state.get('ActualTCPPose', np.zeros(6))
                print(f"Current joints (rad): {current_joints}")
                print(f"Current TCP pose: {current_pose}")
                
                # Use home joint position to determine target pose
                # Calculate TCP pose for home joint configuration, but keep current orientation
                # For UR5/UR10 with home joints [0,-90,-90,-90,90,0] degrees
                # This typically results in TCP position approximately at (x=0.0, y=-0.4, z=0.4)
                home_tcp_position = np.array([0.0, -0.4, 0.4])  # x, y, z from home joints
                
                # Use current pose orientation to maintain robot's current orientation
                current_orientation = current_pose[3:6]  # rx, ry, rz from current pose
                
                # Combine home position with current orientation
                target_pose = np.concatenate([home_tcp_position, current_orientation])
                print(f"Target pose: position from home joints + current orientation: {target_pose}")
                print(f"  Home position: {home_tcp_position}")
                print(f"  Current orientation: {current_orientation}")
                
                # Get current state after pose calculation
                state = env.get_robot_state()
                current_tcp_pose = state['ActualTCPPose']
                print(f"Current robot TCP pose: {current_tcp_pose}")
                print(f"Target pose set to: {target_pose} (based on home joint position)")
                
                # Visual feedback
                t_start = time.monotonic()
                iter_idx = 0
                auto_start_countdown = 5  # seconds
                
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    
                    # pump obs for visualization
                    obs = env.get_obs()
                    
                    # visualize with countdown
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    
                    elapsed_time = time.monotonic() - t_start
                    remaining_time = max(0, auto_start_countdown - elapsed_time)
                    
                    if remaining_time > 0:
                        text = f'Episode: {episode_id} | Auto-start in {remaining_time:.1f}s (Press C to start now, Q to quit)'
                    else:
                        text = f'Episode: {episode_id} | Ready! (Press C to start, Q to quit)'
                    
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c') or remaining_time <= 0:
                        # Start dataset replay immediately or auto-start after countdown
                        print("Starting dataset replay...")
                        break
                    
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=shape_meta)
                            # Convert numpy arrays to tensors
                            obs_dict = {}
                            for key, value in obs_dict_np.items():
                                obs_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        print(f"Action shape: {action.shape}")
                        if action.shape[-1] >= 6:
                            print(f"Joint actions (0:6): {action[:, :6] if action.shape[-1] >= 6 else action}")
                        if action.shape[-1] >= 7:
                            print(f"Gripper actions (6): {action[:, 6]}")
                        
                        # Use action directly as joint commands (6 dimensions)
                        this_target_joints = action[:, :6]  # Use first 6 dimensions as joint commands
                        
                        # No need for delta action processing with joint control
                        # Joint commands are absolute positions

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        
                        # Handle gripper actions timing
                        gripper_actions = None
                        gripper_timestamps = None
                        if action.shape[-1] >= 7:
                            if delta_action:
                                gripper_actions = action[-1][6:7]  # 7th dimension is gripper action
                                gripper_timestamps = action_timestamps[-1:]  # Use last timestamp for delta mode
                            else:
                                gripper_actions = action[:,6:7]  # 7th dimension is gripper action
                                gripper_timestamps = action_timestamps.copy()
                        
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_joints = this_target_joints[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_joints = this_target_joints[is_new]
                            action_timestamps = action_timestamps[is_new]
                            # Update gripper timestamps for valid actions
                            if gripper_actions is not None and gripper_timestamps is not None:
                                gripper_actions = gripper_actions[is_new] if len(gripper_actions) > 1 else gripper_actions
                                gripper_timestamps = gripper_timestamps[is_new] if len(gripper_timestamps) > 1 else gripper_timestamps

                        # Joint angles don't need clipping like pose positions
                        # But we can add safety limits if needed
                        # this_target_joints = np.clip(this_target_joints, joint_min_limits, joint_max_limits)

                        # execute joint actions
                        env.exec_joint_actions(
                            joint_actions=this_target_joints,
                            timestamps=action_timestamps
                        )
                        
                        # execute gripper actions
                        if gripper_actions is not None and gripper_timestamps is not None:
                            for i, (grip_action, grip_timestamp) in enumerate(zip(gripper_actions, gripper_timestamps)):
                                # Clip gripper action to valid range [0,1]
                                gripper_pos = np.clip(grip_action[0], 0.0, 1.0)
                                env.exec_gripper_action(gripper_pos, grip_timestamp)
                                if i == 0:  # Only print for first action to avoid spam
                                    print(f"Gripper command: pos={gripper_pos:.3f}")
                        
                        print(f"Submitted {len(this_target_joints)} joint commands" + 
                              (f" with {len(gripper_actions)} gripper commands" if gripper_actions is not None else "") + ".")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        
                        # Get current episode info from policy
                        episode_info = policy.get_current_episode_info()
                        if episode_info:
                            text = 'Dataset Replay - Episode: {}, Time: {:.1f}, Progress: {}'.format(
                                episode_info['episode_idx'], time.monotonic() - t_start, episode_info['progress']
                            )
                        else:
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                        
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        else:
                            # out of the area
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
