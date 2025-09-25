from typing import Any, Dict, Union
import os

import numpy as np
import sapien
import torch
import trimesh

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickBox-v1")
class PickBoxEnv(BaseEnv):
    """
    **Task Description:**
    A task where the objective is to grasp a box object and return it to the home position.
    The robot must pick up the box and bring it back to where the gripper started.

    **Randomizations:**
    - the box's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the box's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the box is grasped by the robot
    - the robot's TCP is within `goal_thresh` (default 0.025m) euclidean distance of the home position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100]
    box_half_size = 0.02
    goal_thresh = 0.025
    box_spawn_half_size = 0.05
    box_spawn_center = (0, 0)

    def __init__(
            self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, num_envs=1, reconfiguration_freq=None,
            box_obj_path = os.path.join(os.path.dirname(__file__), "assets", "bleach_cleanser", "textured.obj"),
            **kwargs
            ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["xarm6_robotiq"]
        self.box_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.box_spawn_half_size = cfg["cube_spawn_half_size"]
        # self.box_spawn_half_size = 0.5
        self.box_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.box_obj_path = box_obj_path

        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)

    @property
    def _default_sensor_configs(self):
        render_pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        mount_pose = sapien_utils.look_at(eye=[0.1, 0., 0.05], target=[0.0, 0.0, 0.35])
        return [
            CameraConfig("render_camera", render_pose, 256, 256, 1, 0.01, 100),
            CameraConfig(
                uid="mount_camera",
                pose=mount_pose,
                width=256,
                height=256,
                fov= np.pi/2,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["wrist_3_link"]
            )
        ]
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        # return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        render_pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        mount_pose = sapien_utils.look_at(eye=[0.1, 0., 0.05], target=[0.0, 0.0, 0.35])
        return [
            CameraConfig("render_camera", render_pose, 512, 512, 1, 0.01, 100),
            CameraConfig(
                uid="mount_camera",
                pose=mount_pose,
                width=512,
                height=512,
                fov= np.pi/2,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["wrist_3_link"]
            )
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Load box object from assets
        box_obj_path = self.box_obj_path

        mesh = trimesh.load(box_obj_path)

        self.box_true_size = mesh.extents
        initial_z = self.box_true_size[2] / 2.0
        
        # Create box actor from mesh
        builder = self.scene.create_actor_builder()
        phys_mtl = self.scene.sub_scenes[0].create_physical_material(
            static_friction=4.0, dynamic_friction=4.0, restitution=0.0
        )
        builder.add_visual_from_file(box_obj_path)
        builder.add_convex_collision_from_file(box_obj_path, material=phys_mtl, density=100)
        builder.initial_pose = sapien.Pose(p=[0, 0, initial_z])
        self.box = builder.build(name="box")    # with extents 0.1, 0.1, 0.1

        self.distractor_objects = ["bowl", "g_cups"]
        self.no_grasp_objs = []
        for obj in self.distractor_objects:
            obj_builder = self.scene.create_actor_builder()
            obj_path = os.path.join(os.path.dirname(__file__), "assets", obj,"textured.obj")
            obj_true_size = trimesh.load(obj_path).extents

            setattr(self, f"{obj}_true_size", obj_true_size)

            initial_z = obj_true_size[2] / 2.0

            obj_builder.add_visual_from_file(obj_path)
            obj_builder.add_convex_collision_from_file(obj_path, material=phys_mtl, density=100)
            obj_builder.initial_pose = sapien.Pose(p=[0, 0, initial_z])
            self.no_grasp_objs.append(obj_builder.build(name=obj))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        xyz = torch.zeros((b, 3), device=self.device)

        table_half_length = self.table_scene.table_length / 2.0 * 0.6
        table_half_width = self.table_scene.table_width / 4.0 * 0.6

        xy = torch.zeros((b, 1), device=self.device)

        region = [[-table_half_length, -table_half_width], [table_half_length, table_half_width]]
        sampler = randomization.UniformPlacementSampler(
            bounds=region, batch_size=b, device=self.device
        )
        radius = torch.linalg.norm(torch.tensor(self.box_true_size[:2], device=self.device)) + 0.03
        xy = xy + sampler.sample(radius, 100)
        xyz[:, :2] = xy
        xyz[:, 2] = self.box_true_size[2] / 2.0

        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
        self.box.set_pose(Pose.create_from_pq(xyz, qs))

        for obj in self.no_grasp_objs:
            xy = (
                torch.rand((b, 2), device=self.device) * self.box_spawn_half_size * 2
                - self.box_spawn_half_size
            )
            radius = torch.linalg.norm(torch.tensor(getattr(self, f"{obj.name}_true_size")[:2], device=self.device)) + 0.03
            xy = xy + sampler.sample(radius, 100)
            xyz[:, :2] = xy
            xyz[:, 2] = getattr(self, f"{obj.name}_true_size")[2] / 2.0
            delattr(self, f"{obj.name}_true_size")

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            obj.set_pose(Pose.create_from_pq(xyz.clone(), qs))

        self.agent.reset(self.agent.init_keyframe.qpos)
        self.agent.robot.set_pose(self.agent.init_keyframe.pose)

        self.home_tcp_pose = self.agent.tcp.pose.p.clone()

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            home_tcp_pos=self.home_tcp_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.box.pose.raw_pose,
                tcp_to_obj_pos=self.box.pose.p - self.agent.tcp_pose.p,
                tcp_to_home_pos = self.home_tcp_pose - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.box)
        is_robot_static = self.agent.is_static(0.2)
        tcp_to_home_dist = torch.linalg.norm(self.home_tcp_pose - self.agent.tcp_pose.p, axis=1)
        is_at_home = tcp_to_home_dist < self.goal_thresh

        return {
            "success": is_grasped & is_robot_static & is_at_home,
            "is_at_home": is_at_home,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.box.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        tcp_to_home_dist = torch.linalg.norm(
            self.agent.tcp_pose.p - self.home_tcp_pose, axis=1
        )
        return_home_reward = 1 - torch.tanh(5 * tcp_to_home_dist)
        reward += return_home_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids == "panda":
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_at_home"] * is_grasped

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5