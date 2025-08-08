from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform
from scipy.spatial.transform import Rotation

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            # Handle image observations
            if key == 'image' and 'camera_0' in env_obs:
                # Map camera_0 to image
                this_imgs_in = env_obs['camera_0']
            elif key in env_obs:
                this_imgs_in = env_obs[key]
            else:
                print(f"Warning: Image key '{key}' not found in environment observations")
                continue
                
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            if key == 'agent_pos' and shape == (14,):
                # Create 14-dimensional agent_pos vector
                # 6 joint positions + 7 ee pose (3 pos + 4 quat) + 1 gripper
                robot_joint = env_obs.get('robot_joint', np.zeros((2, 6)))  # 6 DOF
                robot_eef_pose = env_obs.get('robot_eef_pose', np.zeros((2, 6)))  # 6D pose (xyz + rxryrz)
                gripper_pos = env_obs.get('gripper_position', np.zeros((2, 1)))  # 1 gripper DOF
                
                # Convert rotation vector to quaternion for 7D ee pose
                t, _ = robot_joint.shape
                agent_pos_list = []
                
                for i in range(t):
                    # 6 joint positions
                    joint_pos = robot_joint[i]  # (6,)
                    
                    # 7D end-effector pose: position (3) + quaternion (4)
                    ee_xyz = robot_eef_pose[i][:3]  # position (3,)
                    ee_rotvec = robot_eef_pose[i][3:]  # rotation vector (3,)
                    ee_quat = Rotation.from_rotvec(ee_rotvec).as_quat()  # quaternion (4,) [x,y,z,w]
                    ee_pose = np.concatenate([ee_xyz, ee_quat])  # (7,)
                    
                    # 1 gripper position
                    gripper = gripper_pos[i]  # (1,)
                    
                    # Concatenate: 6 + 7 + 1 = 14
                    agent_pos = np.concatenate([joint_pos, ee_pose, gripper])  # (14,)
                    agent_pos_list.append(agent_pos)
                
                obs_dict_np[key] = np.stack(agent_pos_list)  # (t, 14)
                print(f"Created agent_pos with shape: {obs_dict_np[key].shape}")
                
            elif key in env_obs:
                this_data_in = env_obs[key]
                if 'pose' in key and shape == (2,):
                    # take X,Y coordinates
                    this_data_in = this_data_in[...,[0,1]]
                obs_dict_np[key] = this_data_in
            else:
                print(f"Warning: Low-dim key '{key}' not found in environment observations")
    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
