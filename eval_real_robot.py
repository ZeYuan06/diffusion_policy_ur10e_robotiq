"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.

================ Joint Control Mode ==============
Use --use_joint_control flag to enable joint angle control instead of pose control.
In joint control mode, the policy outputs 6D joint angles + 1D gripper control.

Automatic Reset:
Use --auto_reset_home flag to automatically reset robot to home position before each episode.
Home position: [0, -90, 90, -90, -90, 0] degrees = [0, -π/2, π/2, -π/2, -π/2, 0] radians

Example usage with joint control:
python eval_real_robot.py -i checkpoint.ckpt -o output_dir --robot_ip 192.168.50.144 --use_joint_control --auto_reset_home
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
import json
import os
from PIL import Image


def process_policy_action(action):
    """
    Process and analyze policy action
    
    Args:
        action: Policy action array
        
    Returns:
        dict: Action information including shape, joint actions, gripper actions
    """
    action_info = {
        'shape': action.shape,
        'joint_actions': action[:, :6] if action.shape[-1] >= 6 else action,
        'has_gripper': action.shape[-1] >= 7,
        'gripper_actions': action[:, 6] if action.shape[-1] >= 7 else None
    }
    return action_info


def process_action_timing(action, obs_timestamp, action_offset, dt, eval_t_start, delta_action):
    """
    Process action timing and filter valid actions
    
    Args:
        action: Policy action array
        obs_timestamp: Latest observation timestamp
        action_offset: Action offset for timing
        dt: Time step
        eval_t_start: Episode start time
        delta_action: Whether using delta actions
        
    Returns:
        dict: Timing information and filtered commands
    """
    # Calculate action timestamps
    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * dt + obs_timestamp
    
    # Check which actions are valid (not too late)
    action_exec_latency = 0.01
    curr_time = time.time()
    is_new = action_timestamps > (curr_time + action_exec_latency)
    
    # Extract joint commands (first 6 dimensions)
    joint_commands = action[:, :6]
    
    # Extract gripper commands if available
    gripper_commands = None
    gripper_timestamps = None
    if action.shape[-1] >= 7:
        if delta_action:
            gripper_commands = action[-1][6:7]  # Last gripper action for delta mode
            gripper_timestamps = action_timestamps[-1:]
        else:
            gripper_commands = action[:, 6:7]  # All gripper actions
            gripper_timestamps = action_timestamps.copy()
    
    # Handle timing budget
    over_budget = False
    budget_info = ""
    
    if np.sum(is_new) == 0:
        # Exceeded time budget, use last action and schedule for next step
        over_budget = True
        joint_commands = joint_commands[[-1]]
        next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
        action_timestamp = eval_t_start + (next_step_idx) * dt
        action_timestamps = np.array([action_timestamp])
        budget_info = f"next action in {action_timestamp - curr_time:.3f}s"
        
        # Update gripper for over budget case
        if gripper_commands is not None:
            gripper_commands = gripper_commands[-1:] if len(gripper_commands) > 1 else gripper_commands
            gripper_timestamps = action_timestamps.copy()
    else:
        # Filter to valid actions
        joint_commands = joint_commands[is_new]
        action_timestamps = action_timestamps[is_new]
        
        # Update gripper timestamps for valid actions
        if gripper_commands is not None and gripper_timestamps is not None:
            gripper_commands = gripper_commands[is_new] if len(gripper_commands) > 1 else gripper_commands
            gripper_timestamps = gripper_timestamps[is_new] if len(gripper_timestamps) > 1 else gripper_timestamps
    
    return {
        'joint_commands': joint_commands,
        'joint_timestamps': action_timestamps,
        'gripper_commands': gripper_commands,
        'gripper_timestamps': gripper_timestamps,
        'over_budget': over_budget,
        'budget_info': budget_info
    }


def execute_gripper_actions(env, gripper_commands, gripper_timestamps):
    """
    Execute gripper actions
    
    Args:
        env: Robot environment
        gripper_commands: Gripper position commands
        gripper_timestamps: Timestamps for gripper commands
    """
    if gripper_commands is not None and gripper_timestamps is not None:
        for i, (grip_action, grip_timestamp) in enumerate(zip(gripper_commands, gripper_timestamps)):
            # Clip gripper action to valid range [0,1]
            gripper_pos = np.clip(grip_action[0], 0.0, 1.0)
            env.exec_gripper_action(gripper_pos, grip_timestamp)
            if i == 0:  # Only print for first action to avoid spam
                print(f"Gripper command: pos={gripper_pos:.3f}")


def debug_policy_inference_and_exit(policy, obs, obs_dict_np, obs_dict, device, output_dir, use_joint_control):
    """
    Debug function to record policy observations and actions, then exit
    
    Args:
        policy: Policy model
        obs: Raw observations from environment
        obs_dict_np: Numpy observation dictionary
        obs_dict: Tensor observation dictionary
        device: PyTorch device
        output_dir: Output directory for saving debug files
        use_joint_control: Whether using joint control mode
    """
    print("="*50)
    print("DEBUG MODE: Recording policy input/output and exiting")
    print("="*50)
    
    # Create debug output directory
    debug_dir = os.path.join(output_dir, 'debug_policy')
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Debug files will be saved to: {debug_dir}")
    
    # 1. Save raw observations
    print("Saving raw observations...")
    obs_save_path = os.path.join(debug_dir, 'raw_observations.json')
    obs_to_save = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_to_save[key] = {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'min': float(np.min(value)) if value.size > 0 else None,
                'max': float(np.max(value)) if value.size > 0 else None,
                'mean': float(np.mean(value)) if value.size > 0 else None
            }
            # Save array data for small arrays (non-image data)
            if value.size < 1000:  # Arbitrary threshold for small arrays
                obs_to_save[key]['data'] = value.tolist()
        else:
            obs_to_save[key] = str(value)
    
    with open(obs_save_path, 'w') as f:
        json.dump(obs_to_save, f, indent=2)
    print(f"Raw observations saved to: {obs_save_path}")
    
    # 2. Save observation images
    print("Saving observation images...")
    image_count = 0
    for key, value in obs.items():
        if 'camera' in key and isinstance(value, np.ndarray):
            # Handle multi-timestep observations
            if len(value.shape) == 4:  # (T, H, W, C)
                for t in range(value.shape[0]):
                    img = value[t]
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    
                    # Convert BGR to RGB if needed
                    if img.shape[-1] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    img_pil = Image.fromarray(img)
                    img_path = os.path.join(debug_dir, f'{key}_t{t}.png')
                    img_pil.save(img_path)
                    image_count += 1
                    print(f"Saved image: {img_path} (shape: {img.shape})")
            elif len(value.shape) == 3:  # (H, W, C)
                img = value
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                
                # Convert BGR to RGB if needed
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_pil = Image.fromarray(img)
                img_path = os.path.join(debug_dir, f'{key}.png')
                img_pil.save(img_path)
                image_count += 1
                print(f"Saved image: {img_path} (shape: {img.shape})")
    
    print(f"Total images saved: {image_count}")
    
    # 3. Save processed observation dictionary
    print("Saving processed observations...")
    obs_dict_save_path = os.path.join(debug_dir, 'processed_observations.json')
    obs_dict_to_save = {}
    for key, value in obs_dict_np.items():
        if isinstance(value, np.ndarray):
            obs_dict_to_save[key] = {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'min': float(np.min(value)) if value.size > 0 else None,
                'max': float(np.max(value)) if value.size > 0 else None,
                'mean': float(np.mean(value)) if value.size > 0 else None
            }
            # Save small arrays
            if value.size < 1000:
                obs_dict_to_save[key]['data'] = value.tolist()
        else:
            obs_dict_to_save[key] = str(value)
    
    with open(obs_dict_save_path, 'w') as f:
        json.dump(obs_dict_to_save, f, indent=2)
    print(f"Processed observations saved to: {obs_dict_save_path}")
    
    # 4. Run policy inference and save actions
    print("Running policy inference...")
    with torch.no_grad():
        policy.reset()
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
    
    print(f"Policy output action shape: {action.shape}")
    print(f"Action range: [{np.min(action):.4f}, {np.max(action):.4f}]")
    
    # 5. Save action data
    print("Saving action data...")
    action_save_path = os.path.join(debug_dir, 'policy_actions.json')
    action_data = {
        'action_shape': list(action.shape),
        'action_dtype': str(action.dtype),
        'action_min': float(np.min(action)),
        'action_max': float(np.max(action)),
        'action_mean': float(np.mean(action)),
        'action_std': float(np.std(action)),
        'full_action_data': action.tolist(),
        'control_mode': 'joint_control' if use_joint_control else 'pose_control'
    }
    
    if action.shape[-1] >= 6:
        action_data['joint_actions'] = action[:, :6].tolist()
        action_data['joint_actions_range'] = [float(np.min(action[:, :6])), float(np.max(action[:, :6]))]
        action_data['joint_actions_mean'] = float(np.mean(action[:, :6]))
    
    if action.shape[-1] >= 7:
        action_data['gripper_actions'] = action[:, 6].tolist()
        action_data['gripper_actions_range'] = [float(np.min(action[:, 6])), float(np.max(action[:, 6]))]
        action_data['gripper_actions_mean'] = float(np.mean(action[:, 6]))
    
    with open(action_save_path, 'w') as f:
        json.dump(action_data, f, indent=2)
    print(f"Action data saved to: {action_save_path}")
    
    # 6. Save action as numpy file for easy loading
    action_np_path = os.path.join(debug_dir, 'policy_actions.npy')
    np.save(action_np_path, action)
    print(f"Action numpy array saved to: {action_np_path}")
    
    # 7. Print summary
    print("\n" + "="*50)
    print("DEBUG SUMMARY")
    print("="*50)
    print(f"Debug directory: {debug_dir}")
    print(f"Raw observations: {obs_save_path}")
    print(f"Processed observations: {obs_dict_save_path}")
    print(f"Images saved: {image_count}")
    print(f"Action data (JSON): {action_save_path}")
    print(f"Action data (NumPy): {action_np_path}")
    print(f"Policy action shape: {action.shape}")
    print(f"Control mode: {'joint_control' if use_joint_control else 'pose_control'}")
    
    if action.shape[-1] >= 6:
        print(f"Joint actions shape: {action[:, :6].shape}")
        print(f"Joint actions range: [{np.min(action[:, :6]):.4f}, {np.max(action[:, :6]):.4f}]")
    
    if action.shape[-1] >= 7:
        print(f"Gripper actions shape: {action[:, 6].shape}")
        print(f"Gripper actions range: [{np.min(action[:, 6]):.4f}, {np.max(action[:, 6]):.4f}]")
    
    print("="*50)
    print("DEBUG MODE COMPLETE - EXITING")
    print("="*50)
    
    # Exit the program
    exit(0)


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--use_joint_control', '-jc', is_flag=True, default=True, help="Use joint control instead of pose control.")
@click.option('--auto_reset_home', '-ar', is_flag=True, default=True, help="Automatically reset robot to home position before each episode.")
@click.option('--debug_mode', '-d', is_flag=True, default=False, help="Enable debug mode: save observations and actions, then exit without robot execution.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, use_joint_control, auto_reset_home, debug_mode):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
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
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                print(f"Warmup action shape: {action.shape}")
                # Action should be at least 2D (position) and may include gripper (7D total for joint control)
                if use_joint_control:
                    assert action.shape[-1] >= 6, f"Expected action dimension >= 6 for joint control, got {action.shape[-1]}"
                    if action.shape[-1] >= 7:
                        print("Policy outputs include joint angles and gripper control")
                    else:
                        print("Policy outputs joint angles only")
                else:
                    assert action.shape[-1] >= 2, f"Expected action dimension >= 2 for pose control, got {action.shape[-1]}"
                    if action.shape[-1] >= 7:
                        print("Policy outputs include pose and gripper control")
                    else:
                        print("Policy outputs pose control only")
                del result
                
                # ========= DEBUG MODE: Save observations and actions then exit ==========
                if debug_mode:
                    print("DEBUG MODE ENABLED - Recording policy data...")
                    debug_policy_inference_and_exit(
                        policy=policy,
                        obs=obs,
                        obs_dict_np=obs_dict_np,
                        obs_dict=obs_dict,
                        device=device,
                        output_dir=output,
                        use_joint_control=use_joint_control
                    )
                    # This function will exit the program, so code below won't execute

            print('Ready!')
            while True:
                # ========= automatic robot reset ==========
                if auto_reset_home:
                    print("Automatic robot reset to home position...")

                    # Define home joint position from RealEnv: [0,-90,90,-90,-90,0] degrees
                    # Converting to radians: [0, -pi/2, pi/2, -pi/2, -pi/2, 0]
                    home_joints_deg = np.array([0, -90, 90, -90, -90, 0])
                    home_joints = home_joints_deg / 180 * np.pi
                    print(f"Home joints (rad): {home_joints}")
                    print(f"Home joints (deg): {home_joints_deg}")
                    
                    # Get current robot state
                    state = env.get_robot_state()
                    current_joints = state.get('ActualQ', np.zeros(6))
                    current_tcp_pose = state.get('ActualTCPPose', np.zeros(6))
                    print(f"Current joints (rad): {current_joints}")
                    print(f"Current TCP pose: {current_tcp_pose}")
                    
                    # Move robot to home joint position directly using joint control
                    print("Moving robot to home joint position...")
                    try:
                        # Use joint control to move to home position
                        # Create a timestamp slightly in the future
                        move_timestamp = time.time() + 1
                        
                        # Execute joint movement to home position
                        env.exec_joint_actions(
                            joint_actions=home_joints.reshape(1, -1),  # Shape: (1, 6)
                            timestamps=np.array([move_timestamp])      # Shape: (1,)
                        )
                        
                        print("Joint movement command sent, waiting for completion...")
                        # Wait for movement to complete
                        time.sleep(3.0)  # Give time for robot to reach home position
                        
                        # Get updated robot state
                        state = env.get_robot_state()
                        current_joints_after = state.get('ActualQ', np.zeros(6))
                        current_tcp_pose_after = state.get('ActualTCPPose', np.zeros(6))
                        print(f"After movement - joints (rad): {current_joints_after}")
                        print(f"After movement - TCP pose: {current_tcp_pose_after}")
                        
                        # Check if we reached the target
                        joint_error = np.abs(current_joints_after - home_joints)
                        max_joint_error = np.max(joint_error)
                        print(f"Joint positioning error: max={max_joint_error:.4f} rad ({max_joint_error*180/np.pi:.2f} deg)")
                        
                        if max_joint_error < 0.05:  # Within 3 degrees
                            print("Successfully moved to home position!")
                        else:
                            print("Warning: Robot may not have reached exact home position")
                            
                    except Exception as e:
                        print(f"Failed to move to home position: {e}")
                        print("Continuing with current position...")
                
                # ========= automatic robot control loop ==========
                print("Starting automated policy control...")
                
                # Visual feedback with countdown
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
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)
                    
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
                        # Start policy control immediately or auto-start after countdown
                        print("Starting policy control...")
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
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        action_info = process_policy_action(action)
                        
                        if use_joint_control:
                            # Use joint control mode
                            print(f"Action shape: {action.shape}")
                            if action.shape[-1] >= 6:
                                print(f"Joint actions: {action[:, :6]}")
                            if action.shape[-1] >= 7:
                                print(f"Gripper actions: {action[:, 6]}")
                            
                            # Handle action timing and filtering
                            timing_info = process_action_timing(
                                action, obs_timestamps[-1], action_offset, dt, 
                                eval_t_start, delta_action
                            )
                            
                            joint_commands = timing_info['joint_commands']
                            joint_timestamps = timing_info['joint_timestamps']
                            gripper_commands = timing_info['gripper_commands']
                            gripper_timestamps = timing_info['gripper_timestamps']

                            # Execute joint actions
                            env.exec_joint_actions(
                                joint_actions=joint_commands,
                                timestamps=joint_timestamps
                            )
                            
                            # Execute gripper actions
                            execute_gripper_actions(env, gripper_commands, gripper_timestamps)
                            
                            # Print execution summary
                            summary = f"Submitted {len(joint_commands)} joint commands"
                            if gripper_commands is not None and len(gripper_commands) > 0:
                                summary += f" with {len(gripper_commands)} gripper commands"
                            summary += "."
                            print(summary)
                        
                        else:
                            # Use original pose control mode
                            print(f"Action shape: {action.shape}")
                            if action.shape[-1] >= 6:
                                print(f"Pose actions: {action[:, :6] if action.shape[-1] >= 6 else action}")
                            if action.shape[-1] >= 7:
                                print(f"Gripper actions: {action[:, 6]}")
                            
                            # Get current robot state for pose control
                            current_robot_state = env.get_robot_state()
                            current_tcp_pose = current_robot_state['TargetTCPPose']
                            
                            if delta_action:
                                assert len(action) == 1
                                if perv_target_pose is None:
                                    perv_target_pose = obs['robot_eef_pose'][-1]
                                this_target_pose = perv_target_pose.copy()
                                this_target_pose[[0,1]] += action[-1][:2]  # Only use first 2 dimensions for position
                                perv_target_pose = this_target_pose
                                this_target_poses = np.expand_dims(this_target_pose, axis=0)
                            else:
                                this_target_poses = np.zeros((len(action), len(current_tcp_pose)), dtype=np.float64)
                                this_target_poses[:] = current_tcp_pose
                                this_target_poses[:,[0,1]] = action[:,:2]  # Use first 2 dimensions for position control
                                this_target_poses[:,[0,1]] = action[:,:2]  # Use first 2 dimensions for position control

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
                                this_target_poses = this_target_poses[[-1]]
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]
                                # Update gripper timestamps for valid actions
                                if gripper_actions is not None and gripper_timestamps is not None:
                                    gripper_actions = gripper_actions[is_new] if len(gripper_actions) > 1 else gripper_actions
                                    gripper_timestamps = gripper_timestamps[is_new] if len(gripper_timestamps) > 1 else gripper_timestamps

                            # clip actions
                            this_target_poses[:,:2] = np.clip(
                                this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                            # execute actions
                            env.exec_actions(
                                actions=this_target_poses,
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
                            
                            print(f"Submitted {len(this_target_poses)} steps of actions" + 
                                  (f" with {len(gripper_actions)} gripper commands" if gripper_actions is not None else "") + ".")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
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
