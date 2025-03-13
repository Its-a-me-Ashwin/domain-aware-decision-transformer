#!/usr/bin/env python3

"""
collect_expert_data.py

Script that collects expert demonstration data using an IK-based
pick-and-place policy for the PickAndPlaceEnv environment.

Requires:
    pick_and_place_env.py in the same directory (or properly installed).
"""

import numpy as np
import pybullet as p
import math
import random
import json
import time

# Import the environment
from env import PickAndPlaceEnv
from scipy.spatial.transform import Rotation as R

gui = True

############################
# HELPER FUNCTIONS
############################

import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_ik(robot_id, end_effector_link_index, target_pos, target_orn):
    """
    Computes the analytical inverse kinematics solution for the Franka Panda robot
    based on the paper: "Analytical Inverse Kinematics for Franka Emika Panda – a Geometrical Solver"
    
    Parameters:
    - robot_id: int (Unused, maintained for interface compatibility)
    - end_effector_link_index: int (Unused, maintained for interface compatibility)
    - target_pos: (3,) -> Desired end-effector position [x, y, z]
    - target_orn: (4,) -> Desired end-effector orientation [quaternion: x, y, z, w]
    
    Returns:
    - np.array of shape (7,) containing the computed joint angles or None if IK fails.
    """

    # Franka Emika Panda DH Parameters
    d1, d3, d5, d7 = 0.333, 0.316, 0.384, 0.107 + 0.1034  # Link offsets
    a4 = 0.0825  # Link length
    alpha = np.pi / 4  # End-effector offset

    # Convert quaternion to rotation matrix
    R_EE = R.from_quat(target_orn).as_matrix()  # (3x3 rotation matrix)
    
    # Compute wrist center (O6) from the end-effector position
    p_EE = np.array(target_pos)  # (3,)
    p_O7 = p_EE - d7 * R_EE[:, 2]  # Move back along the Z-axis of the end-effector
    p_O6 = p_O7  # No additional offset needed

    # Compute shoulder-to-wrist distance
    p_O2 = np.array([0, 0, d1])  # Fixed base joint position
    O2O6 = np.linalg.norm(p_O6 - p_O2)  # Distance between O2 and O6

    # Check reachability condition
    if not (np.abs(d3 + d5) >= O2O6 >= np.abs(d3 - d5)):
        print("IK Failure: Target position is out of reach")
        return None

    # Step 1: Solve for q1 (Base rotation)
    q1 = np.arctan2(p_O6[1], p_O6[0])  # Rotation about base

    # Step 2: Solve for q3 using triangle law
    cos_q3 = (O2O6**2 - d3**2 - d5**2) / (2 * d3 * d5)
    q3 = np.arccos(np.clip(cos_q3, -1, 1))  # Ensure valid value

    # Step 3: Solve for q2 (vertical motion)
    shoulder_to_wrist_vec = p_O6 - p_O2
    shoulder_proj = np.linalg.norm(shoulder_to_wrist_vec[:2])  # XY projection
    q2 = np.arctan2(shoulder_to_wrist_vec[2], shoulder_proj)

    # Step 4: Solve for q4 (elbow configuration)
    q4 = -q3  # Ensures proper folding

    # Step 5: Compute wrist rotation for q5, q6, q7
    R3_6 = np.linalg.inv(R.from_euler('zxy', [q1, q2, q3]).as_matrix()) @ R_EE
    q5 = np.arctan2(R3_6[1, 2], R3_6[0, 2])
    q6 = np.arctan2(R3_6[2, 0], -R3_6[2, 1])
    q7 = np.arctan2(R3_6[2, 2], -R3_6[1, 2])

    # Assemble the final joint solution
    q_solution = np.array([q1, q2, q3, q4, q5, q6, q7])

    return q_solution  # Matches the interface of the original function


def move_arm(env, target_joint_positions, steps=500, gui=gui):
    """
    Moves the robot arm from current joint positions to the target positions
    in a linear interpolation. Collects data along the way.
    
    target_joint_positions: (7,)
    """
    # Ensure we have a NumPy array of shape (7,)
    target_joint_positions = np.array(target_joint_positions, dtype=np.float32)  # (7,)

    # Current joint states:
    current_joint_states = p.getJointStates(env.robot, range(7)) # each is (position, velocity, ...)
    # current_joints => shape (7,)
    current_joints = np.array([s[0] for s in current_joint_states], dtype=np.float32)  # (7,)

    # Linear interpolation from current_joints to target_joint_positions
    trajectory = []
    for t in range(1, steps + 1):
        alpha = t / steps
        # Both current_joints and target_joint_positions are (7,)
        interp = (1 - alpha) * current_joints + alpha * target_joint_positions  # (7,)
        trajectory.append(interp)

    # Step through each sub-step in the environment
    data = []
    for joint_positions in trajectory:  # joint_positions => shape (7,)
        obs, reward, done, info = env.step(joint_positions)  # env.step expects (7,)
        data.append({
            'obs': obs.copy(),
            'action': joint_positions.copy(),
            'reward': reward,
            'done': done
        })
        if gui:
            time.sleep(0.05)
        if done:
            break

    return data

def pick_and_place_expert(env):
    """
    1) Moves above the cube
    2) Lowers to grasp
    3) Lifts
    4) Moves over the goal
    5) Lowers and releases (conceptually, though release not coded in env)
    6) Lifts up slightly
    Returns a list of data from all steps.
    """

    # For Franka Panda in PyBullet, link index 11 or 8 is often the end-effector.
    # But let's guess or adjust to your URDF (we'll try 11).
    end_effector_index = 11

    ###########
    # Step 1) Move above the cube
    ###########
    obj_pos, obj_orn = p.getBasePositionAndOrientation(env.object)
    # We'll place the end-effector 10cm above the object's center
    approach_height = 0.05
    above_cube_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + approach_height]
    # Orientation: let’s keep a vertical downward orientation in quaternions
    down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])  # point Z- down

    # Solve IK
    q_above = compute_ik(env.robot, end_effector_index, above_cube_pos, down_orientation)

    data = []
    data.extend(move_arm(env, q_above, steps=100))

    ###########
    # Step 2) Lower to grasp
    ###########
    # grasp_height_offset = 0.01  # near the surface
    # grasp_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + grasp_height_offset]
    # q_grasp = compute_ik(env.robot, end_effector_index, grasp_pos, down_orientation)
    # data.extend(move_arm(env, q_grasp, steps=100))

    # # (Here you would close gripper if your robot has one. We ignore for simplicity.)

    # ###########
    # # Step 3) Lift up the cube
    # ###########
    # lift_height = 0.15
    # lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + lift_height]
    # q_lift = compute_ik(env.robot, end_effector_index, lift_pos, down_orientation)
    # data.extend(move_arm(env, q_lift, steps=100))

    # ###########
    # # Step 4) Move to goal above
    # ###########
    # goal_pos = env.config.goal_position
    # approach_goal_height = 0.10
    # above_goal_pos = [goal_pos[0], goal_pos[1], goal_pos[2] + approach_goal_height]
    # q_above_goal = compute_ik(env.robot, end_effector_index, above_goal_pos, down_orientation)
    # data.extend(move_arm(env, q_above_goal, steps=150))

    # ###########
    # # Step 5) Lower object onto the goal
    # ###########
    # place_height_offset = 0.01
    # place_pos = [goal_pos[0], goal_pos[1], goal_pos[2] + place_height_offset]
    # q_place = compute_ik(env.robot, end_effector_index, place_pos, down_orientation)
    # data.extend(move_arm(env, q_place, steps=100))

    # # (Open gripper here if you have one.)

    # ###########
    # # Step 6) Lift up slightly again
    # ###########
    # q_lift_after = compute_ik(env.robot, end_effector_index, above_goal_pos, down_orientation)
    # data.extend(move_arm(env, q_lift_after, steps=100))

    return data

############################
# MAIN SCRIPT
############################

def main():
    num_episodes = 1  # or however many you like
    all_data = []      # will store all episodes

    for ep in range(num_episodes):
        # Randomize domain parameters
        gravity = random.uniform(-15.0, -5.0)
        friction = random.uniform(0.1, 1.0)
        mass = random.uniform(0.2, 2.0)
        wind = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)]
        # Random goal in front of the robot
        gx = random.uniform(0.5, 0.75)
        gy = random.uniform(-0.2, 0.2)  # slight lateral
        gz = 0.05

        config_dict = {
            'gravity': gravity,
            'object_friction': friction,
            'object_mass': mass,
            'wind_force': wind,
            'goal_position': [gx, gy, gz],
            'orientation_threshold': 0.1,
        }

        # Create environment with these parameters (headless for speed)
        env = PickAndPlaceEnv(config_dict=config_dict, gui=gui)
        env.reset()

        # Collect expert data
        trajectory_data = pick_and_place_expert(env)

        # Optionally store domain parameters in each data point
        # Or store them once as metadata:
        episode_data = {
            'domain_params': config_dict,
            'trajectory': trajectory_data
        }
        all_data.append(episode_data)

        print(f"Episode {ep+1}/{num_episodes} collected, success steps = {len(trajectory_data)}")

        env.close()

    # Save all data to a file
    # For instance as a .npz or .json. Here, let's do .npz for compactness.
    np.savez('expert_demonstrations.npz', demos=all_data)
    print("Data saved to expert_demonstrations.npz")


if __name__ == "__main__":
    main()
