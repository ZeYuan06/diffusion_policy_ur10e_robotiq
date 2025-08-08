#!/usr/bin/env python3
"""
测试夹爪集成的脚本 - 与eval_real_robot.py保持一致的配置
"""

import sys
import tempfile
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.real_env import RealEnv

def test_gripper_integration():
    """测试夹爪集成是否正常工作 - 使用与eval_real_robot.py相同的配置"""
    print("Testing Gripper Integration...")
    
    # 配置参数 - 与eval_real_robot.py保持一致
    robot_ip = "192.168.1.11"  # 请根据实际情况修改
    frequency = 10
    n_obs_steps = 2  # 默认值
    obs_res = (640, 480)  # 默认观测分辨率
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with SharedMemoryManager() as shm_manager:
                # 创建环境 - 使用与eval_real_robot.py相同的参数
                env = RealEnv(
                    output_dir=temp_dir,
                    robot_ip=robot_ip,
                    frequency=frequency,
                    n_obs_steps=n_obs_steps,
                    obs_image_resolution=obs_res,
                    obs_float32=True,  # 与eval_real_robot.py一致
                    init_joints=False,
                    enable_multi_cam_vis=True,  # 启用多相机可视化
                    record_raw_video=True,     # 启用视频录制
                    thread_per_video=3,        # 与eval_real_robot.py一致
                    video_crf=21,              # 与eval_real_robot.py一致
                    use_gripper=True,
                    gripper_port=63352,
                    shm_manager=shm_manager
                )
                
                print("Starting environment...")
                env.start(wait=True)
                
                if env.is_ready:
                    print("Environment ready!")
                    
                    # 设置相机参数 - 与eval_real_robot.py一致
                    print("Setting camera parameters...")
                    env.realsense.set_exposure(exposure=120, gain=0)
                    env.realsense.set_white_balance(white_balance=5900)
                    
                    print("Waiting for realsense...")
                    time.sleep(1.0)
                    
                    # 调试RTDE观测
                    print("\n=== 调试RTDE观测 ===")
                    debug_data = env.debug_rtde_observations()
                    
                    # 获取观测数据
                    print("Getting observations...")
                    obs = env.get_obs()
                    print(f"\n=== 观测数据键 ===")
                    for key in obs.keys():
                        if isinstance(obs[key], np.ndarray):
                            print(f"{key}: shape={obs[key].shape}, dtype={obs[key].dtype}")
                        else:
                            print(f"{key}: {type(obs[key])}")
                    
                    # 检查机器人状态
                    print("\n=== 机器人状态 ===")
                    state = env.get_robot_state()
                    if 'TargetTCPPose' in state:
                        target_pose = state['TargetTCPPose']
                        print(f"Target TCP Pose: {target_pose}")
                    
                    # 测试夹爪控制
                    if 'gripper_position' in obs:
                        print(f"\n=== 夹爪控制测试 ===")
                        current_pos = obs['gripper_position'][-1]
                        print(f"Current gripper position: {current_pos}")
                        
                        # 测试夹爪张开
                        print("Testing gripper open (position 0.2)...")
                        env.command_gripper(0.2, speed=255, force=100)
                        time.sleep(2)  # 等待足够时间
                        
                        # 获取新状态
                        obs = env.get_obs()
                        new_pos = obs['gripper_position'][-1]
                        print(f"New gripper position: {new_pos}")
                        
                        # 测试夹爪关闭
                        print("Testing gripper close (position 0.8)...")
                        env.command_gripper(0.8, speed=255, force=100)
                        time.sleep(2)  # 等待足够时间
                        
                        # 获取最终状态
                        obs = env.get_obs()
                        final_pos = obs['gripper_position'][-1]
                        print(f"Final gripper position: {final_pos}")
                        
                        # 恢复到中间位置
                        print("Returning to middle position (0.5)...")
                        env.command_gripper(0.5, speed=255, force=100)
                        time.sleep(2)
                        
                        obs = env.get_obs()
                        final_pos = obs['gripper_position'][-1]
                        print(f"Final gripper position: {final_pos}")
                        
                    else:
                        print("Warning: gripper_position not found in observations!")
                        print("Available keys:", list(obs.keys()))
                    
                    print("\n=== 测试完成 ===")
                    print("夹爪集成测试成功完成!")
                    
                else:
                    print("Environment not ready!")
                    
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print("Stopping environment...")
            if 'env' in locals():
                env.stop(wait=True)
            print("Environment stopped.")

def test_minimal():
    """最小化测试 - 用于调试相机问题"""
    print("Testing minimal configuration...")
    robot_ip = "192.168.1.11"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with SharedMemoryManager() as shm_manager:
                # 最小化配置
                env = RealEnv(
                    output_dir=temp_dir,
                    robot_ip=robot_ip,
                    frequency=10,
                    camera_serial_numbers=['dummy_camera'],  # 使用虚拟相机
                    enable_multi_cam_vis=False,              # 禁用可视化
                    record_raw_video=False,                  # 禁用录制
                    use_gripper=True,
                    gripper_port=63352,
                    shm_manager=shm_manager
                )
                
                print("Starting minimal environment...")
                env.start(wait=True)
                
                if env.is_ready:
                    print("Minimal environment ready!")
                    debug_data = env.debug_rtde_observations()
                    print("Minimal test completed successfully!")
                else:
                    print("Minimal environment not ready!")
                    
                env.stop(wait=True)
                
        except Exception as e:
            print(f"Minimal test failed: {e}")
            import traceback
            traceback.print_exc()

def test_standalone():
    """独立测试RTDE观测 - 使用最小配置"""
    print("Testing standalone RTDE observations...")
    robot_ip = "192.168.1.11"
    
    try:
        data = RealEnv.debug_rtde_standalone(
            robot_ip=robot_ip,
            use_gripper=True,
            gripper_port=63352,
            camera_serial_numbers=['dummy_camera'],  # 使用虚拟相机避免检测问题
            enable_multi_cam_vis=False,
            record_raw_video=False
        )
        print("Standalone test completed successfully!")
        return data
    except Exception as e:
        print(f"Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "standalone":
    #         test_standalone()
    #     elif sys.argv[1] == "minimal":
    #         test_minimal()
    #     elif sys.argv[1] == "full":
    #         test_gripper_integration()
    #     else:
    #         print("Usage: python test_gripper_integration.py [full|minimal|standalone]")
    #         print("  full: 完整测试，模拟eval_real_robot.py的配置")
    #         print("  minimal: 最小化测试，禁用相机功能")
    #         print("  standalone: 独立RTDE测试")
    # else:
    #     print("选择测试模式:")
    #     print("1. 完整测试 (推荐，模拟eval_real_robot.py)")
    #     print("2. 最小化测试 (如果有相机问题)")
    #     print("3. 独立测试 (仅测试RTDE)")
        
    #     choice = input("请输入选择 (1/2/3): ").strip()
    #     if choice == "1":
    #         test_gripper_integration()
    #     elif choice == "2":
    #         test_minimal()
    #     elif choice == "3":
    #         test_standalone()
    #     else:
    #         print("无效选择，运行完整测试...")
    test_gripper_integration()
