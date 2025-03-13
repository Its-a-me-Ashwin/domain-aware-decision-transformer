# pick_and_place_env.py

import pybullet as p
import pybullet_data
import numpy as np
import time
import json
import random
from scipy.spatial.transform import Rotation as R

class EnvConfig:
    def __init__(self, config_path=None, config_dict=None):
        """
        Modified constructor so we can either load from a JSON file
        or directly from a dictionary (config_dict).
        """
        if config_dict is not None:
            config = config_dict
        else:
            config = None
            if config_path is not None:
                try:
                    with open(config_path, 'r') as file:
                        config = json.load(file)
                except (FileNotFoundError, FileExistsError):
                    print("Using Default configs")
            if config is None:
                config = {}

        self.object_mass = config.get('object_mass', 1.0)
        self.object_friction = config.get('object_friction', 0.5)
        self.gravity = config.get('gravity', -9.8)
        self.wind_force = config.get('wind_force', [0.0, 0.0, 0.0])

        # Robot base at (0,0,0) on the plane
        self.robot_base_position = [0, 0, 0]

        # Object starts near the robot, slightly above plane
        self.object_start_position = [0.25, 0.25, 0.05]
        self.object_start_orientation = [0, 0, 0, 1]

        # Random goal between 50cm and 75cm in front of the robot
        # (But we can override this externally if needed)
        self.goal_position = [random.uniform(0.5, 0.75), random.uniform(0.5, 0.75), 0.05]
        self.goal_orientation = [0, 0, 0, 1]
        self.orientation_threshold = config.get('orientation_threshold', 0.1)


class PickAndPlaceEnv:
    def __init__(self, config_path=None, config_dict=None, gui=False):
        self.config = EnvConfig(config_path=config_path, config_dict=config_dict)
        self.gui = gui
        self.dt = 1/240

        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)

        # Load environment
        self._load_environment()

    def _load_environment(self):
        """Load the plane, robot, object, and target indicator."""
        self.plane = p.loadURDF("plane.urdf")

        self.robot = p.loadURDF("franka_panda/panda.urdf",
                                basePosition=self.config.robot_base_position,
                                useFixedBase=True)

        # Fix robot base (optional; some URDFs are already fixed)
        p.createConstraint(
            parentBodyUniqueId=self.robot, parentLinkIndex=-1,
            childBodyUniqueId=-1, childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=self.config.robot_base_position,
            childFramePosition=self.config.robot_base_position
        )

        self.object = p.loadURDF("cube_small.urdf",
                                 basePosition=self.config.object_start_position,
                                 baseOrientation=self.config.object_start_orientation)

        # Adjust mass, friction
        p.changeDynamics(self.object, -1,
                         mass=self.config.object_mass,
                         lateralFriction=self.config.object_friction)
        # Color the object red
        p.changeVisualShape(self.object, -1, rgbaColor=[1, 0, 0, 1])

        # Create target object
        self.target_object = p.loadURDF("cube_small.urdf",
                                        basePosition=self.config.goal_position,
                                        baseOrientation=self.config.goal_orientation)

        # Make the target static
        p.changeDynamics(self.target_object, -1, mass=0)
        p.setCollisionFilterGroupMask(self.target_object, -1, 0, 0)
        # Color the target green
        p.changeVisualShape(self.target_object, -1, rgbaColor=[0, 1, 0, 1])

        # Apply wind force if any
        if any(self.config.wind_force):
            p.applyExternalForce(
                self.object,                  # objectUniqueId
                -1,                           # linkIndex
                self.config.wind_force,       # forceObj
                self.config.object_start_position,  # posObj
                p.WORLD_FRAME                 # flags
            )

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, self.config.gravity)
        self._load_environment()
        return self.get_observations()

    def get_observations(self):
        pos, orn = p.getBasePositionAndOrientation(self.object)
        joint_states = p.getJointStates(self.robot, range(7))
        joint_positions = [state[0] for state in joint_states]
        return np.concatenate([joint_positions, pos, orn])

    def compute_reward(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object)
        pos_error = np.linalg.norm(np.array(obj_pos) - np.array(self.config.goal_position))

        # Quaternion distance (orientation error)
        obj_rot = R.from_quat(obj_orn)
        goal_rot = R.from_quat(self.config.goal_orientation)
        orn_error = R.inv(obj_rot) * goal_rot
        orn_error_angle = orn_error.magnitude()

        reward = - (pos_error + orn_error_angle)
        return reward

    def step(self, action):
        """action is array of length 7 for the robot arm joints (ignoring gripper here)."""
        for j in range(7):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, action[j])
        p.stepSimulation()

        obs = self.get_observations()
        reward = self.compute_reward()
        done = self._check_done()
        return obs, reward, done, {}

    def _check_done(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object)
        pos_error = np.linalg.norm(np.array(obj_pos) - np.array(self.config.goal_position))

        obj_rot = R.from_quat(obj_orn)
        goal_rot = R.from_quat(self.config.goal_orientation)
        orn_error = R.inv(obj_rot) * goal_rot
        orn_error_angle = orn_error.magnitude()

        if pos_error < 0.05 and orn_error_angle < self.config.orientation_threshold:
            return True
        return False

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = PickAndPlaceEnv(config_path='config.json', gui=True)
    obs = env.reset()
    
    for _ in range(2000):
        #action = np.random.uniform(-0.2, 0.2, 7)  # Random joint actions
        action = np.array([0, 0, 0, 0, 0, 3.14, 0])
        obs, reward, done, _ = env.step(action)
        if done:
            print("Goal reached!")
            break
        time.sleep(0.01)

    env.close()
