import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random

from math import pi

class InvertedPendulumBalanceEnv(gym.Env):
    """
    A Gym-like environment for balancing a double inverted pendulum on a cart using PyBullet.
    This environment includes a reward function and a 'done' condition so it can be used with an RL algorithm.
    """

    def __init__(self,
                 urdf_path="./singlePendulum.urdf",
                 use_gui=False,
                 time_step=1./240.,
                 max_force=50.0,
                 random_offset=0.1,
                 random_cart_offset=0.1,
                 max_episode_steps=10_000):
        """
        Args:
            urdf_path (str): Path to the URDF file.
            use_gui (bool): Whether to use p.GUI (rendered) or p.DIRECT (headless).
            time_step (float): The physics timestep used in stepSimulation.
            max_force (float): Maximum force/torque applied to the prismatic joint (action).
            random_offset (float): Magnitude of the random offset for initial pendulum angles.
            max_episode_steps (int): Episode will end if we exceed this many steps.
        """
        super().__init__()

        self.urdf_path = urdf_path
        self.use_gui = use_gui
        self.time_step = time_step
        self.max_force = max_force
        self.random_offset = random_offset
        self.random_cart_offset = random_cart_offset
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Connect to PyBullet
        if self.use_gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        # Configure PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -10, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        # Load plane and pendulum
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.startPos = [0, 0, 1]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.box_id = p.loadURDF(
            self.urdf_path,
            self.startPos,
            self.startOrientation,
            useFixedBase=False,
            physicsClientId=self.client_id
        )
        self.num_joints = p.getNumJoints(self.box_id, physicsClientId=self.client_id)

        # Disable default motor control (so pendulums are free)
        for j in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.box_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client_id
            )

        # We'll store old velocities to compute accelerations
        self.prev_joint_vel = np.zeros(self.num_joints)

        # ---------------------------
        # Define Action/Observation Spaces
        # ---------------------------
        # Action: 1D (force on the prismatic joint), range [-max_force, max_force]
        self.action_space = spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        # Observation: we will have 9 elements:
        #   [cart_x, cart_x_dot, cart_x_acc,
        #    p1_theta, p1_theta_dot, p1_theta_acc,
        #    p2_theta, p2_theta_dot, p2_theta_acc]
        # We'll pick some large bounding box for each dimension.
        high = np.array([np.finfo(np.float32).max]*6, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, shape=(6,), dtype=np.float32)

    def reset(self, seed=0):
        """Reset the environment and return the initial observation."""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -10, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.box_id = p.loadURDF(
            self.urdf_path,
            self.startPos,
            self.startOrientation,
            useFixedBase=False,
            physicsClientId=self.client_id
        )
        self.num_joints = p.getNumJoints(self.box_id, physicsClientId=self.client_id)

        for j in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.box_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client_id
            )

        # Slight random offsets in pendulum angles (joint 1 & 2)
        offset1 = random.uniform(-self.random_offset, self.random_offset) + pi
        offset2 = random.uniform(-self.random_cart_offset, self.random_cart_offset)

        # Joint 0: prismatic (cart), Joint 1: pend1, Joint 2: pend2
        p.resetJointState(self.box_id, 0, offset2, 0.0, physicsClientId=self.client_id)
        p.resetJointState(self.box_id, 1, offset1, 0.0, physicsClientId=self.client_id)

        # Reset velocity cache
        self.prev_joint_vel = np.zeros(self.num_joints)

        # Reset step count
        self.current_step = 0

        return self._get_observation(), dict()

    def step(self, action):
        """Take an action, step the simulation, and return (obs, reward, done, info)."""
        self.current_step += 1
        action = action[0] * self.max_force
        # Clip the force to the valid range
        force = np.clip(action, -self.max_force, self.max_force)

        # Apply force to prismatic joint (joint 0)
        p.setJointMotorControl2(
            bodyUniqueId=self.box_id,
            jointIndex=0,
            controlMode=p.TORQUE_CONTROL,  # linear force for prismatic
            force=force,
            physicsClientId=self.client_id
        )

        # Step simulation
        p.stepSimulation(physicsClientId=self.client_id)

        # Get obs
        obs = self._get_observation()

        # Parse out angles, cart x, etc. from obs
        cart_x = obs[0]
        p1_theta = obs[3] % (2*pi)
        p1_theta_dot = obs[4]
        # ---------------------------
        # Reward Function
        # ---------------------------
        # We want both pendulums upright => p1_theta ~ 0, p2_theta ~ 0
        # Also encourage cart_x ~ 0.
        # We'll penalize squared angles & squared cart position.
        # Optionally, we can penalize large forces to encourage minimal control.
        angle_penalty = 5*(p1_theta-pi)**2
        velocity_penalty = 0.5 * ((p1_theta_dot**2))
        cart_penalty = 0.5*(cart_x**2)
        force_penalty = 0.1*(force**2)

        angle_upright_bonus = 100 if abs(p1_theta-pi) < 0.2 else 0
        cart_bonus = 1 if abs(cart_x) < 1 else 0 
        # ---------------------------
        # Done Condition
        # ---------------------------
        # Example: If cart goes out of [-4, 4], or pendulums fall past +/- 1 rad
        # or we exceed max_episode_steps
        done = self.current_step >= self.max_episode_steps

        # Reward is negative sum of penalties, so the agent tries to minimize them.
        reward = - (angle_penalty + velocity_penalty + cart_penalty + force_penalty) + angle_upright_bonus + cart_bonus
        if abs(cart_x) > 10.0:
            reward -= 1000
        if abs(p1_theta-pi) > pi/2:
            pass
            #reward -= 10000
        
        info = {}

        return obs, reward, done, False, info

    def _get_observation(self):
        """
        Return:
          [cart_x, cart_x_dot, cart_x_acc,
           p1_theta, p1_theta_dot, p1_theta_acc,
           p2_theta, p2_theta_dot, p2_theta_acc]
        """
        states = [
            p.getJointState(self.box_id, j, physicsClientId=self.client_id)
            for j in range(self.num_joints)
        ]
        positions = [s[0] for s in states]  # (cart_x, p1_theta, p2_theta)
        velocities = [s[1] for s in states]  # (cart_x_dot, p1_dot, p2_dot)

        # Accelerations
        accelerations = []
        for j in range(self.num_joints):
            acc_j = (velocities[j] - self.prev_joint_vel[j]) / self.time_step
            accelerations.append(acc_j)

        # Update stored velocities
        self.prev_joint_vel = np.array(velocities)

        obs = np.array([
            positions[0],           # cart_x
            velocities[0],          # cart_x_dot
            accelerations[0],       # cart_x_acc
            positions[1] % (2*pi),  # p1_theta
            velocities[1],          # p1_theta_dot
            accelerations[1],       # p1_theta_acc
        ], dtype=np.float32)

        return obs

    def render(self, mode="human"):
        """PyBullet does the rendering automatically if p.GUI is used."""
        pass

    def close(self):
        p.disconnect(physicsClientId=self.client_id)

if __name__ == "__main__":
    env = InvertedPendulumBalanceEnv(use_gui=True, random_offset=0.1, random_cart_offset=2)
    obs, _ = env.reset()
    for _ in range(10000):
        action = [np.random.uniform(-1, 1)]
        obs, reward, done, _, info = env.step(action)
        if done:
            obs, _ = env.reset()
        #print("Obs:", (57.3*obs[3]) % (360) - 180, "Reward:", reward)
        print(reward)
        time.sleep(1./240.)
    env.close()
