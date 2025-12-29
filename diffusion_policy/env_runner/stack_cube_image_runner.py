"""
UR10e + Robotiq PickCube Image Runner
Ported from diffusion policy runner, adapted for ManiSkill with Parallel Support
"""

from pathlib import Path
from typing import Dict
import numpy as np
import gymnasium as gym
import torch
import pathlib
import collections
import tqdm
import wandb
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_euler_angles,
)
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from .pyroki_env import PickBoxEnv
from .robotiq_ur10e.ur10e_robotiq import UR10eRobotiq


def process_batch_obs(raw_obs, env_unwrapped, device, dtype):
    processed_obs = {}

    # --- Process Images ---
    # raw_obs['sensor_data'][cam]['rgb'] is [B, H, W, C] (uint8 or float)
    # Policy expects [B, C, H, W] (float 0-1)

    def process_img(img_tensor):
        if isinstance(img_tensor, torch.Tensor):
            # [B, H, W, C] -> [B, C, H, W]
            img = img_tensor.permute(0, 3, 1, 2).float() / 255.0
        else:
            # Fallback for numpy (shouldn't happen in vectorized mode usually)
            img = torch.from_numpy(img_tensor).permute(0, 3, 1, 2).float() / 255.0
        return img.to(device=device, dtype=dtype)

    processed_obs["base_image"] = process_img(
        raw_obs["sensor_data"]["base_camera"]["rgb"]
    )
    processed_obs["wrist_image"] = process_img(
        raw_obs["sensor_data"]["wrist_camera"]["rgb"]
    )

    # --- Process Proprioception ---

    # TCP Pose [B, 3] and [B, 4]
    tcp_pose = env_unwrapped.agent.tcp_pose
    tcp_pos = tcp_pose.p
    tcp_quat = tcp_pose.q

    # Convert Quaternion to Euler (XYZ)
    rotation_matrix = quaternion_to_matrix(tcp_quat)
    euler_angles = matrix_to_euler_angles(rotation_matrix, convention="XYZ")

    # [B, 6] -> [x, y, z, r, p, y]
    ee_pose_6d = torch.cat([tcp_pos, euler_angles], dim=1)

    # Gripper [B, 1] (approximate, depends on gripper definition)
    # Using the last dim of qpos as per your reference code
    gripper_pos = env_unwrapped.agent.robot.get_qpos()[:, -1:]

    if len(gripper_pos.shape) == 1:
        gripper_pos = gripper_pos.unsqueeze(1)

    # Concatenate: [B, 3+3+1] = [B, 7]
    agent_pos = torch.cat([ee_pose_6d, gripper_pos], dim=1)
    processed_obs["agent_pos"] = agent_pos.to(device=device, dtype=dtype)

    return processed_obs


class StackCubeImageRunner(BaseImageRunner):
    """UR10e + Robotiq StackCube image runner using ManiSkill"""

    def __init__(
        self,
        output_dir,
        env_id=PickBoxEnv.uid,
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        n_test=5,
        vis_test=True,
        test_start_seed=10000,
        max_steps=200,
        n_obs_steps=8,
        tqdm_interval_sec=5.0,
    ):
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
        self.vis_test = vis_test
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.tqdm_interval_sec = tqdm_interval_sec

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
        n_envs = self.n_test

        assets_dir = Path(__file__).parent / "assets" / "YCB_processed"
        all_obj_paths = []
        for item in assets_dir.iterdir():
            if item.is_dir():
                obj_file = item / "textured.obj"
                if obj_file.exists():
                    all_obj_paths.append(str(obj_file))

        box_obj_path = str(assets_dir / "005_tomato_soup_can" / "textured.obj")
        env = gym.make(
            self.env_id,
            num_envs=n_envs,
            obs_mode="state_dict+rgb",
            robot_uids=UR10eRobotiq.uid,
            box_obj_path=box_obj_path,
            distractor_pool_paths=all_obj_paths,
            render_mode=self.render_mode,
            control_mode=self.control_mode,
            max_episode_steps=self.max_steps,
        )

        if self.vis_test:
            video_dir = self.output_dir / "media"
            video_dir.mkdir(parents=True, exist_ok=True)
            env = RecordEpisode(
                env,
                output_dir=str(video_dir),
                save_video=True,
                max_steps_per_video=self.max_steps,
                # We rely on ManiSkill to handle which envs to record.
                # Usually it records all if not specified, or we can filter later.
            )

        # Reset Environment
        seeds = [self.test_start_seed + i for i in range(n_envs)]
        obs, info = env.reset(seed=seeds)

        # Initialize Observation History
        # obs_deque will contain dictionaries where values are [B, ...] tensors
        obs_deque = collections.deque(maxlen=self.n_obs_steps)
        # Initial process
        batch_obs = process_batch_obs(obs, env.unwrapped, device, dtype)
        for _ in range(self.n_obs_steps):
            obs_deque.append(batch_obs)

        # Metrics Tracking
        # Track max reward/success per environment index
        max_rewards = torch.zeros(n_envs, device=device)
        success_tracker = torch.zeros(n_envs, device=device, dtype=torch.bool)

        # Action buffer for Open-Loop execution
        # We plan every n_action_steps
        cached_actions = None

        # 4. Main Evaluation Loop
        pbar = tqdm.tqdm(
            range(self.max_steps),
            desc=f"Eval Parallel ({n_envs} envs)",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )
        step_in_plan = 0
        cached_actions = None

        for step in pbar:
            # Policy Inference
            # Replan if we ran out of actions or haven't planned yet
            if cached_actions is None or step_in_plan >= cached_actions.shape[1]:
                    # Stack observations: [B, T, ...]
                    policy_input = {}
                    for key in batch_obs.keys():
                        # Stack along time dimension (dim 1)
                        # obs_deque[t][key] is [B, ...] -> stack -> [B, T, ...]
                        policy_input[key] = torch.stack(
                            [x[key] for x in obs_deque], dim=1
                        )

                    with torch.no_grad():
                        action_dict = policy.predict_action(policy_input)
                        # [B, Ta, Action_Dim]
                        cached_actions = action_dict["action"]

                    # Reset execution counter
                    step_in_plan = 0

            # Execute Action
            action_to_step = cached_actions[:, step_in_plan]
            obs, reward, terminated, truncated, info = env.step(action_to_step)
            step_in_plan += 1

            # Update Metrics
            current_rewards = reward.to(device)

            # Update max reward seen so far
            max_rewards = torch.maximum(max_rewards, current_rewards)

            # Update success
            if "success" in info:
                is_success = info["success"]
                if isinstance(is_success, torch.Tensor):
                    is_success = is_success.to(device)
                else:
                    is_success = torch.tensor(is_success, device=device)
                success_tracker = success_tracker | is_success

            # --- Update Observation ---
            batch_obs = process_batch_obs(obs, env.unwrapped, device, dtype)
            obs_deque.append(batch_obs)

            if step % 10 == 0:
                pbar.set_postfix(
                    {
                        "reward": f"{max_rewards.mean().item():.2f}",
                        "success": f"{success_tracker.float().mean().item():.1%}",
                    }
                )

        pbar.close()
        env.close()

        # Logging and Results
        log_data = dict()

        # Aggregate metrics
        mean_reward = max_rewards.mean().item()
        success_rate = success_tracker.float().mean().item()

        print(
            f"Eval Complete: Mean Reward: {mean_reward:.3f}, Success Rate: {success_rate:.1%}"
        )

        log_data["test/mean_score"] = mean_reward
        log_data["test/success_rate"] = success_rate

        # Log individual env metrics
        for i in range(n_envs):
            seed = self.test_start_seed + i
            log_data[f"test/sim_max_reward_{seed}"] = max_rewards[i].item()
            log_data[f"test/sim_success_{seed}"] = float(success_tracker[i].item())

        # Upload Videos
        if self.vis_test and wandb.run is not None:
            video_dir = self.output_dir / "media"
            video_files = sorted(
                list(video_dir.glob("*.mp4")), key=lambda x: x.stat().st_mtime
            )

            if len(video_files) > 0:
                # The last file is the most recent one
                video_file = video_files[-1]

                if video_file.stat().st_size > 0:
                    # Upload the single grid video
                    wandb.log(
                        {
                            "test/video_grid": wandb.Video(
                                str(video_file),
                                caption=f"Parallel Eval ({n_envs} envs)",
                            )
                        },
                        step=current_step,
                    )
                    print(f"Uploaded video grid: {video_file.name}")

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
                env = self.create_single_env(
                    record_video=(i < 2), video_filename=f"test_{seed}.mp4"
                )
            except Exception as e:
                print(f"Error creating environment for random test: {e}")
                results.append(
                    {
                        "seed": seed,
                        "steps": 0,
                        "total_reward": 0.0,
                        "success": False,
                        "error": f"Environment creation failed: {str(e)}",
                    }
                )
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
                    "seed": seed,
                    "steps": steps_executed,
                    "total_reward": total_reward,
                    "success": total_reward > 0.1,  # Simple success criterion
                }

                results.append(result)

                if debug:
                    print(
                        f"Episode {i+1}: {steps_executed} steps, reward: {total_reward:.3f}"
                    )
                else:
                    status = "✅" if result["success"] else "❌"
                    print(f"Test {i+1}: {status} (reward: {total_reward:.3f})")

            except Exception as e:
                print(f"Error in random test episode {i+1}: {e}")
                results.append(
                    {
                        "seed": seed,
                        "steps": 0,
                        "total_reward": 0.0,
                        "success": False,
                        "error": str(e),
                    }
                )
            finally:
                try:
                    env.close()
                except Exception as e:
                    print(f"Error closing random test environment: {e}")

        # Summary
        success_rate = np.mean([r["success"] for r in results])
        avg_reward = np.mean([r["total_reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])

        print(f"\n=== Random Action Test Summary ===")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Average steps: {avg_steps:.1f}")

        return {
            "results": results,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        }
