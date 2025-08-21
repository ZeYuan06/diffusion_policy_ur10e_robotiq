from copy import deepcopy

import numpy as np
import sapien
import torch
from typing import List, Optional, Union

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import assets, download_asset, sapien_utils
from mani_skill import format_path
import os

# A utility function to deepcopy a dictionary containing controller configs
def deepcopy_dict(x: dict):
    y = dict()
    for k, v in x.items():
        if isinstance(v, dict):
            y[k] = deepcopy_dict(v)
        else:
            y[k] = deepcopy(v)
    return y

@register_agent()
class UR10eRobotiq(BaseAgent):
    """
    UR10e arm with a Robotiq 2F-85 gripper.
    The model is a custom MJCF file combining the UR10e arm and the Robotiq gripper.
    Controller logic is adapted from the XArm6Robotiq agent.
    """
    uid = "ur10e_robotiq"
    # Use the local URDF file
    urdf_path = os.path.join(os.path.dirname(__file__), "ur10e_robotiq.urdf")

    # Specify material and contact patch parameters for Robotiq gripper
    urdf_config = dict(
        _materials = dict(
            # Material named "gripper" with high friction and no elasticity
            gripper = dict(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0
            )
        ),
        link = dict(
            # Assign the same gripper material to left and right inner finger pads and set patch radius
            left_inner_finger_pad = dict(
                material="gripper",
                patch_radius=0.1,
                min_patch_radius=0.1
            ),
            right_inner_finger_pad = dict(
                material="gripper",
                patch_radius=0.1,
                min_patch_radius=0.1
            ),
        ),
    )

    # Define the joint names for the UR10e arm
    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Controller parameters adapted from XArm6Robotiq
    arm_stiffness = 5000
    arm_damping = 100
    arm_force_limit = 330 # Taken from UR10e defaults

    gripper_stiffness = 1e4
    gripper_damping = 1e3
    gripper_force_limit = 100
    
    # The end-effector link for EE-space controllers
    ee_link_name = "robotiq_arg2f_base_link"

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm Controllers
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=None, upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names,
            lower=-0.3, upper=0.3,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper Controllers
        # -------------------------------------------------------------------------- #
        # The two main finger joints that will be actively controlled
        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]

        # The other finger joints are passive and are driven by equality constraints in the MJCF
        passive_finger_joint_names = [
            "left_inner_knuckle_joint", "right_inner_knuckle_joint",
            "left_inner_finger_joint", "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names, damping=0, friction=0
        )

        # A mimic controller allows a single action to control both main finger joints
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=0.0, upper=0.81, # Gripper open/close range
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Combined Controllers
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=finger_mimic_pd_joint_pos,
                passive=passive_finger_joints,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=finger_mimic_pd_joint_pos, # Gripper usually takes absolute commands
                passive=passive_finger_joints,
            ),
        )
        # Add EE-space controllers
        for ee_type in ["pos", "pose"]:
            if ee_type == "pos":
                arm_ee_cfg = PDEEPosControllerConfig(
                    joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
                    stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit,
                    ee_link=self.ee_link_name,
                )
            else:
                arm_ee_cfg = PDEEPoseControllerConfig(
                    joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1, rot_lower=-0.1, rot_upper=0.1,
                    stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit,
                    ee_link=self.ee_link_name,
                )
            controller_configs[f"pd_ee_{ee_type}"] = dict(
                arm=arm_ee_cfg, gripper=finger_mimic_pd_joint_pos, passive=passive_finger_joints
            )
            controller_configs[f"pd_ee_delta_{ee_type}"] = dict(
                arm=arm_ee_cfg, gripper=finger_mimic_pd_joint_pos, passive=passive_finger_joints
            )

        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        # NOTE: The scene.create_drive part from XArm6 is removed as the gripper's
        # closed-loop mechanism is now handled by <equality> constraints in the MJCF file.
        
        # Disable self-collisions within the gripper and between the gripper and the last arm link.
        gripper_links = [
            "robotiq_arg2f_base_link",
            "left_outer_knuckle", "left_inner_knuckle", "left_outer_finger", "left_inner_finger",
            "right_outer_knuckle", "right_inner_knuckle", "right_outer_finger", "right_inner_finger",
            "wrist_3_link",  # Last link of the UR10e arm
        ]
        
        for link_name in gripper_links:
            if link_name in self.robot.links_map:
                link = self.robot.links_map[link_name]
                link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        # Find the finger and TCP links for helper functions
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        if self.tcp is None:
            raise ValueError(f"Link with name {self.ee_link_name} not found.")

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        # This function is copied from XArm6Robotiq and should work if links are found
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        return torch.logical_and(lforce > min_force, rforce > min_force)

    @property
    def tcp_pose(self):
        return self.tcp.pose
    
    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """
        Loads the robot articulation
        """

        def build_articulation(scene_idxs: Optional[List[int]] = None):
            if self.urdf_path is not None:
                loader = self.scene.create_urdf_loader()
                asset_path = format_path(str(self.urdf_path))
            elif self.mjcf_path is not None:
                loader = self.scene.create_mjcf_loader()
                asset_path = format_path(str(self.mjcf_path))

            loader.name = self.uid
            if self._agent_idx is not None:
                loader.name = f"{self.uid}-agent-{self._agent_idx}"
            loader.fix_root_link = self.fix_root_link
            loader.load_multiple_collisions_from_file = self.load_multiple_collisions
            loader.disable_self_collisions = self.disable_self_collisions

            if self.urdf_config is not None:
                urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
                sapien_utils.check_urdf_config(urdf_config)
                sapien_utils.apply_urdf_config(loader, urdf_config)

            if not os.path.exists(asset_path):
                print(f"Robot {self.uid} definition file not found at {asset_path}")
                if (
                    self.uid in assets.DATA_GROUPS
                    or len(assets.DATA_GROUPS[self.uid]) > 0
                ):
                    response = download_asset.prompt_yes_no(
                        f"Robot {self.uid} has assets available for download. Would you like to download them now?"
                    )
                    if response:
                        for (
                            asset_id
                        ) in assets.expand_data_group_into_individual_data_source_ids(
                            self.uid
                        ):
                            download_asset.download(assets.DATA_SOURCES[asset_id])
                    else:
                        print(
                            f"Exiting as assets for robot {self.uid} are not downloaded"
                        )
                        exit()
                else:
                    print(
                        f"Exiting as assets for robot {self.uid} are not found. Check that this agent is properly registered with the appropriate download asset ids"
                    )
                    exit()
            builders, _, _ = loader.parse(asset_path)
            builder = builders[0]
            # builder.initial_pose = initial_pose
            if scene_idxs is not None:
                builder.set_scene_idxs(scene_idxs)
                builder.set_name(f"{self.uid}-agent-{self._agent_idx}-{scene_idxs}")
            initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            robot = builder.build(self.scene, initial_pose)
            assert robot is not None, f"Fail to load URDF/MJCF from {asset_path}"
            
            # Create a simple wrapper class instead of using full Articulation
            class SimpleArticulationWrapper:
                def __init__(self, articulation):
                    self._articulation = articulation
                    self.links_map = {link.name: link for link in articulation.get_links()}
                    
                def get_links(self):
                    return self._articulation.get_links()
                    
                def get_active_joints(self):
                    return self._articulation.get_active_joints()
                    
                def get_active_joint_names(self):
                    return [joint.name for joint in self._articulation.get_active_joints()]
            
            return SimpleArticulationWrapper(robot)

        if self.build_separate:
            arts = []
            for scene_idx in range(self.scene.num_envs):
                robot = build_articulation([scene_idx])
                self.scene.remove_from_state_dict_registry(robot._articulation)
                arts.append(robot._articulation)
            # For simplicity, we'll use the first articulation for testing
            class SimpleArticulationWrapper:
                def __init__(self, articulation):
                    self._articulation = articulation
                    self.links_map = {link.name: link for link in articulation.get_links()}
                    
                def get_links(self):
                    return self._articulation.get_links()
                    
                def get_active_joints(self):
                    return self._articulation.get_active_joints()
                    
                def get_active_joint_names(self):
                    return [joint.name for joint in self._articulation.get_active_joints()]
            
            self.robot = SimpleArticulationWrapper(arts[0])
            self.scene.add_to_state_dict_registry(arts[0])
        else:
            self.robot = build_articulation()
        # Cache robot link names
        self.robot_link_names = [link.name for link in self.robot.get_links()]

@register_agent()
class UR10eRobotiqWristCamera(UR10eRobotiq):
    uid = "ur10e_robotiq_wristcam"

    @property
    def _sensor_configs(self):
        # Mounts a camera to the gripper's base.
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0.08], q=sapien.Pose.from_axis_angle([0, 1, 0], -np.pi/2).q),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["robotiq_arg2f_base_link"],
            )
        ]
    
if __name__ == "__main__":
    # Create a minimal scene for testing
    scene = sapien.Scene()
    scene.set_timestep(1 / 240.0)
    
    # Initialize the agent with required parameters
    agent = UR10eRobotiq(scene=scene, control_freq=20)
    print("active joints:", agent.robot.get_active_joint_names())