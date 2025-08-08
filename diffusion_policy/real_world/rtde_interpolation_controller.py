import os
import time
import enum
import socket
import threading
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from typing import OrderedDict, Tuple, Union
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class RobotiqGripper:
    """Communicates with the gripper directly, via socket with string commands, leveraging string names for variables."""

    # WRITE VARIABLES (CAN ALSO READ)
    ACT = "ACT"  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = "GTO"  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = "ATR"  # atr : auto-release (emergency slow move)
    ADR = "ADR"  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = "FOR"  # for : force (0-255)
    SPE = "SPE"  # spe : speed (0-255)
    POS = "POS"  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = "STA"  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = "PRE"  # position request (echo of last commanded position)
    OBJ = "OBJ"  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = "FLT"  # fault (0=ok, see manual for errors if not zero)

    ENCODING = "UTF-8"  # ASCII and UTF-8 both seem to work

    class GripperStatus(enum.Enum):
        """Gripper status reported by the gripper."""
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(enum.Enum):
        """Object status reported by the gripper."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 10.0) -> None:
        """Connects to a gripper at the given address."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert self.socket is not None
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        if self.socket is not None:
            self.socket.close()

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """Sends the appropriate command via socket to set the value of n variables."""
        assert self.socket is not None
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += "\n"
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends command to set the value of a variable."""
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Retrieves the value of a variable from the gripper."""
        assert self.socket is not None
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data.decode(self.ENCODING)}: does not match '{variable}'")
        return int(value_str)

    @staticmethod
    def _is_ack(data: bytes):
        return data == b"ack"

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.POS)

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position."""
        position = int(position)
        speed = int(speed)
        force = int(force)

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        var_dict: OrderedDict[str, Union[int, float]] = OrderedDict([
            (self.POS, clip_pos),
            (self.SPE, clip_spe),
            (self.FOR, clip_for),
            (self.GTO, 1),
        ])
        succ = self._set_vars(var_dict)
        time.sleep(0.008)  # need to wait
        return succ, clip_pos


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    GRIPPER_MOVE = 3
    SCHEDULE_JOINT_WAYPOINT = 4


class RTDEInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            use_gripper=True,
            gripper_port=63352,
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        use_gripper: whether to use gripper
        gripper_port: gripper communication port

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.use_gripper = use_gripper  # 现在总是可用，因为我们集成了RobotiqGripper类
        self.gripper_port = gripper_port

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_joints': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'gripper_pos': 0.0, 
            'gripper_speed': 100.0,
            'gripper_force': 50.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get'+key)())
        example['robot_receive_timestamp'] = time.time()
        
        # Add gripper state to examples
        if self.use_gripper:
            example['gripper_position'] = np.array([0.0])  # gripper position [0-1]
            example['gripper_force'] = np.array([0.0])     # gripper force
            example['gripper_speed'] = np.array([0.0])     # gripper speed
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration,
            'target_time': 0.0,
            'gripper_pos': 0.0, 
            'gripper_speed': 100.0,  
            'gripper_force': 50.0 
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        """
        Pose: 6D end-effector pose
        target_time: absolute time in seconds
        """
        assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time,
            'duration': 0.1, 
            'gripper_pos': 0.0,
            'gripper_speed': 100.0,
            'gripper_force': 50.0,
        }
        self.input_queue.put(message)

    def schedule_joint_waypoint(self, joints, target_time):
        """
        Schedule joint waypoint for execution
        
        Args:
            joints: 6D joint angles in radians
            target_time: absolute time in seconds
        """
        assert self.is_alive()
        joints = np.array(joints)
        assert joints.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_JOINT_WAYPOINT.value,
            'target_joints': joints,
            'target_time': target_time,
            'duration': 0.1,
            'gripper_pos': 0.0,
            'gripper_speed': 100.0,
            'gripper_force': 50.0,
        }
        self.input_queue.put(message)
    
    def command_gripper(self, gripper_pos, speed=255, force=100):
        """Command gripper to move to specified position
        
        Args:
            gripper_pos: gripper position [0-1], 0=fully open, 1=fully closed
            speed: gripper speed [0-255]
            force: gripper force [0-255]
        """
        if not self.use_gripper:
            return
            
        assert 0 <= gripper_pos <= 1, "Gripper position must be between 0 and 1"
        message = {
            'cmd': Command.GRIPPER_MOVE.value,
            'target_pose': np.zeros(6),
            'gripper_pos': gripper_pos,
            'gripper_speed': speed,
            'gripper_force': force,
            'target_time': 0.0,
            'duration': 0.0
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip)
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        
        # Initialize gripper connection
        gripper = None
        if self.use_gripper:
            try:
                gripper = RobotiqGripper()
                gripper.connect(hostname=robot_ip, port=self.gripper_port)
                if self.verbose:
                    print(f"[RTDEPositionalController] Gripper connected on port {self.gripper_port}")
            except Exception as e:
                print(f"Warning: Failed to connect gripper: {e}")
                self.use_gripper = False
                gripper = None

        def get_gripper_state():
            """Helper function to get gripper state"""
            if gripper is not None:
                try:
                    import time
                    time.sleep(0.001)  # Small delay to avoid too frequent queries
                    pos = gripper.get_current_position()
                    # Map 0-255 to 0-1
                    normalized_pos = pos / 255.0 if pos is not None else 0.0
                    # Can add more state information
                    return {
                        'gripper_position': np.array([normalized_pos]),
                        'gripper_force': np.array([0.0]),
                        'gripper_speed': np.array([0.0])
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to get gripper state: {e}")
                    return {
                        'gripper_position': np.array([0.0]),
                        'gripper_force': np.array([0.0]),
                        'gripper_speed': np.array([0.0])
                    }
            else:
                return {
                    'gripper_position': np.array([0.0]),
                    'gripper_force': np.array([0.0]),
                    'gripper_speed': np.array([0.0])
                }

        try:
            if self.verbose:
                print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                rtde_c.setTcp(self.tcp_offset_pose)
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
                else:
                    assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency
            curr_pose = rtde_r.getActualTCPPose()
            curr_joints = rtde_r.getActualQ()  # Get current joint positions
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            # Create joint interpolator for joint control
            joint_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_joints]
            )
            
            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()

                # send command to robot
                t_now = time.monotonic()

                # Use joint control mode
                joint_command = joint_interp(t_now)
                vel = [1.05] * 6  # Joint velocities for each joint
                acc = [1.4] * 6   # Joint accelerations for each joint
                # Use servoJ for joint control
                assert rtde_c.servoJ(joint_command, vel, acc, dt)
                
                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get'+key)())
                state['robot_receive_timestamp'] = time.time()
                
                # Add gripper state
                if self.use_gripper:
                    gripper_state = get_gripper_state()
                    state.update(gripper_state)
                
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.SCHEDULE_JOINT_WAYPOINT.value:
                        target_joints = command['target_joints']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        
                        # Use joint interpolator for smooth joint movement
                        joint_interp = joint_interp.schedule_waypoint(
                            pose=target_joints,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                        if self.verbose:
                            print(f"[RTDEPositionalController] New joint target: {target_joints} at time: {target_time}")
                    elif cmd == Command.GRIPPER_MOVE.value:
                        if gripper is not None:
                            try:
                                gripper_pos = int(float(command['gripper_pos']) * 255)  # 转换到0-255范围并确保为int
                                gripper_speed = int(command.get('gripper_speed', 255))
                                gripper_force = int(command.get('gripper_force', 100))
                                gripper.move(gripper_pos, gripper_speed, gripper_force)
                                if self.verbose:
                                    print(f"[RTDEPositionalController] Gripper move to {gripper_pos}/255")
                            except Exception as e:
                                if self.verbose:
                                    print(f"Warning: Gripper move failed: {e}")
                    else:
                        keep_running = False
                        break

                # regulate frequency
                rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            rtde_c.servoStop()

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
