"""
测试夹爪集成的脚本
"""

import sys
import tempfile
import time
import numpy as np
from diffusion_policy.real_world.real_env import RealEnv

def test_gripper_integration():
    """测试夹爪集成是否正常工作"""
    print("Testing Gripper Integration...")
    
    # 配置参数
    robot_ip = "192.168.1.11"  # 请根据实际情况修改
    
    # 创建临时输出目录
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 创建环境
            env = RealEnv(
                output_dir=temp_dir,
                robot_ip=robot_ip,
                use_gripper=True,
                gripper_port=63352,
                frequency=10,
                camera_serial_numbers=[],  # 不使用相机进行测试
                enable_multi_cam_vis=False
            )
            
            print("Starting environment...")
            env.start(wait=True)
            
            if env.is_ready:
                print("Environment ready!")
                
                # 调试RTDE观测
                print("\n=== 调试RTDE观测 ===")
                debug_data = env.debug_rtde_observations()
                
                # 检查夹爪状态是否在观测中
                obs = env.get_obs()
                print(f"\n=== 观测数据键 ===")
                for key in obs.keys():
                    if isinstance(obs[key], np.ndarray):
                        print(f"{key}: shape={obs[key].shape}, dtype={obs[key].dtype}")
                    else:
                        print(f"{key}: {type(obs[key])}")
                
                # 测试夹爪控制
                if 'gripper_position' in obs:
                    print(f"\n=== 夹爪控制测试 ===")
                    current_pos = obs['gripper_position'][-1]
                    print(f"Current gripper position: {current_pos}")
                    
                    # 测试夹爪张开
                    print("Testing gripper open (position 0.2)...")
                    env.command_gripper(0.2, speed=255, force=100)
                    time.sleep(1)
                    
                    # 获取新状态
                    obs = env.get_obs()
                    new_pos = obs['gripper_position'][-1]
                    print(f"New gripper position: {new_pos}")
                    
                    # 测试夹爪关闭
                    print("Testing gripper close (position 0.8)...")
                    env.command_gripper(0.8, speed=255, force=100)
                    time.sleep(1)
                    
                    # 获取最终状态
                    obs = env.get_obs()
                    final_pos = obs['gripper_position'][-1]
                    print(f"Final gripper position: {final_pos}")
                    
                else:
                    print("Warning: gripper_position not found in observations!")
                
            else:
                print("Environment not ready!")
                
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            print("Stopping environment...")
            env.stop(wait=True)
            print("Environment stopped.")

def test_standalone():
    """独立测试RTDE观测"""
    print("Testing standalone RTDE observations...")
    robot_ip = "192.168.1.11"  # 请根据实际情况修改
    
    try:
        data = RealEnv.debug_rtde_standalone(
            robot_ip=robot_ip,
            use_gripper=True,
            gripper_port=63352,
            camera_serial_numbers=[],
            enable_multi_cam_vis=False
        )
        print("Standalone test completed successfully!")
        return data
    except Exception as e:
        print(f"Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "standalone":
        test_standalone()
    else:
        test_gripper_integration()
