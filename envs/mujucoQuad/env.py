#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barkour_randomized.py

An all-in-one script that:
1) Defines a QuadWorldConfig for domain randomization.
2) Defines the BarkourEnv with domain randomization.
3) Uses Brax PPO to train and collect data, saving config and dataset.
"""

import os
import json
import torch
import jax
import jax.numpy as jp
from jax import Array
import numpy as np
import secrets
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Brax / ML specifics
import brax
from brax.io import mjcf
from brax import envs
from brax import base
#from brax.envs import register
from brax.envs.base import PipelineEnv, State
from brax.base import Motion, Transform
from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.agents.ppo.train import train as ppo_train

# Checkpointing
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Plotting & progress
import matplotlib.pyplot as plt
from tqdm import tqdm
import mediapy as media


def plot_training_metrics_separately(logs_dict, save_path=None):
    print(logs_dict)
    for label, log in logs_dict.items():
        if not log:
            print(f"Skipping data for {label} – it’s empty.")
            continue

        # Prepare the data
        steps, values = zip(*log)

        # Create a new figure for each metric
        plt.figure()
        plt.plot(steps, values, marker="o", label=label)

        # Annotate each data point with its y-value
        for step, val in zip(steps, values):
            plt.annotate(
                f"{val:.2f}",                # Format to 2 decimal places
                xy=(step, val),             # The point to label
                xytext=(5, 5),             # Offset text slightly from the point
                textcoords="offset points"  # Interpret xytext as offset from xy
            )

        plt.xlabel("Training Steps")
        plt.ylabel(label)
        plt.title(f"Training Metric: {label}")
        plt.grid(True)
        plt.legend()

        # Optionally save each figure
        if save_path:
            # e.g., save the figure as "<save_path>_loss.png", "<save_path>_reward.png"
            metric_save_path = f"{save_path}_{label}.png"
            plt.savefig(metric_save_path)
            print(f"[✓] Saved training plot for '{label}' to {metric_save_path}")

        plt.show()

# -------------------------------------------------------------------
# 1. Domain Randomization Class
# -------------------------------------------------------------------
class QuadWorldConfig:
    def __init__(
        self,
        gravity_strength: float = 9.81,
        gravity_angle_x: float = 0.0,
        gravity_angle_y: float = 0.0,
        max_motor_torque: float = 35.0,
        friction: float = 1.0,
        incline_x: float = 0.0,
        incline_y: float = 0.0,
    ):
        self.gravity_strength = gravity_strength
        self.gravity_angle_x = gravity_angle_x
        self.gravity_angle_y = gravity_angle_y
        self.max_motor_torque = max_motor_torque
        self.friction = friction
        self.incline_x = incline_x
        self.incline_y = incline_y

    @staticmethod
    def generate_random_config():
        """Create a config with random gravity, friction, torque, etc."""
        return QuadWorldConfig(
            gravity_strength=np.random.uniform(8.0, 12.0),
            gravity_angle_x=np.random.uniform(-0.5, 0.5),
            gravity_angle_y=np.random.uniform(-0.5, 0.5),
            max_motor_torque=np.random.uniform(20.0, 50.0),
            friction=np.random.uniform(0.2, 2.0),
            incline_x=np.random.uniform(-5.0, 5.0),
            incline_y=np.random.uniform(-5.0, 5.0),
        )

    def to_dict(self) -> Dict[str, float]:
        """For saving as JSON."""
        return {
            "gravity_strength": float(self.gravity_strength),
            "gravity_angle_x": float(self.gravity_angle_x),
            "gravity_angle_y": float(self.gravity_angle_y),
            "max_motor_torque": float(self.max_motor_torque),
            "friction": float(self.friction),
            "incline_x": float(self.incline_x),
            "incline_y": float(self.incline_y),
        }

    def config_to_vector(self) -> List[float]:
        """Vector encoding for logging or storing per step."""
        return [
            self.gravity_strength,
            self.gravity_angle_x,
            self.gravity_angle_y,
            self.max_motor_torque,
            self.friction,
            self.incline_x,
            self.incline_y,
        ]

# -------------------------------------------------------------------
# 2. Barkour Environment with Domain Randomization
# -------------------------------------------------------------------
from ml_collections import config_dict
# We'll also import mjx for the backend
from brax.mjx.base import State as MjxState

#from brax.mjx.base import PipelineEnv as MjxPipelineEnv  # not strictly needed

import mujoco
from mujoco import mjx as mjx_backend
from brax.io import model

from etils import epath

BARKOUR_ROOT_PATH = epath.Path('mujoco_menagerie/google_barkour_vb')

def get_config():
    """Returns base reward config for barkour quadruped environment."""
    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        # Basic tracking
                        tracking_lin_vel=1.5,
                        tracking_ang_vel=0.8,
                        # Regularizations
                        lin_vel_z=-2.0,
                        ang_vel_xy=-0.05,
                        orientation=-5.0,
                        torques=-0.0002,
                        action_rate=-0.01,
                        feet_air_time=0.2,
                        stand_still=-0.5,
                        termination=-1.0,
                        foot_slip=-0.1,
                    )
                ),
                tracking_sigma=0.25,
            )
        )
        return default_config

    return config_dict.ConfigDict(dict(rewards=get_default_rewards_config()))


class BarkourEnv(PipelineEnv):
    """Environment for training the Barkour quadruped with randomizable physics."""

    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        scene_file: str = 'scene_mjx.xml',
        config: Optional[QuadWorldConfig] = None,
        seed: int = 0,
        **kwargs,
    ):
        # 1) Load MJCF
        path = BARKOUR_ROOT_PATH / scene_file
        sys = mjcf.load(path.as_posix())

        self._dt = 0.02  # 50 fps
        sys = sys.tree_replace({'opt.timestep': 0.004})

        # 2) Override some defaults
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        # 3) Domain Randomization using config
        if config is None:
            config = QuadWorldConfig()
        # friction
        geom_friction = np.tile([config.friction, 0.005, 0.0001], (sys.ngeom, 1))
        sys = sys.replace(geom_friction=geom_friction)
        # gravity
        g_vec = np.array([
            config.gravity_angle_x,
            config.gravity_angle_y,
            -1.0
        ])
        g_vec = g_vec / np.linalg.norm(g_vec) * config.gravity_strength
        sys = sys.replace(opt=sys.opt.replace(gravity=g_vec))
        # max motor torque
        sys = sys.replace(
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(config.max_motor_torque)
        )
        # incline
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('xy', [config.incline_x, config.incline_y], degrees=True)
        quat = rot.as_quat()  # [x, y, z, w]
        sys = sys.replace(body_quat=sys.body_quat.at[0].set(quat[[3, 0, 1, 2]]))

        # 4) Final call to super
        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        # 5) Setup rest
        self.reward_config = get_config()
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso')
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jp.array([0.52, 2.1, 2.1] * 4)

        feet_site = [
            'foot_front_left',
            'foot_hind_left',
            'foot_front_right',
            'foot_hind_right',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        if any(id_ == -1 for id_ in feet_site_id):
            raise ValueError("Foot site not found.")
        self._feet_site_id = np.array(feet_site_id)

        lower_leg_body = [
            'lower_leg_front_left',
            'lower_leg_hind_left',
            'lower_leg_front_right',
            'lower_leg_hind_right',
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        if any(id_ == -1 for id_ in lower_leg_body_id):
            raise ValueError("Body not found.")
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Random linear and angular velocities in XY-plane + yaw."""
        lin_vel_x = [-0.6, 1.5]
        lin_vel_y = [-0.8, 0.8]
        ang_vel_yaw = [-0.7, 0.7]
        _, key1, key2, key3 = jax.random.split(rng, 4)
        vx = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        vy = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        vz = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
        return jp.array([vx[0], vy[0], vz[0]])

    def reset(self, rng: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'last_vel': jp.zeros(12),
            'command': self.sample_command(cmd_rng),
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            'kick': jp.array([0.0, 0.0]),
            'step': 0,
        }

        obs_history = jp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = 0.0
        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

        # Kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt

        # done check
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(brax.math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = {
            'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
            'tracking_ang_vel': self._reward_tracking_ang_vel(state.info['command'], x, xd),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(state.info['command'], joint_angles),
            'feet_air_time': self._reward_feet_air_time(state.info['feet_air_time'], first_contact, state.info['command']),
            'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 1e4)

        # update state info
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jp.where(
            state.info['step'] > 500, self.sample_command(cmd_rng), state.info['command']
        )
        # reset step counter when done
        state.info['step'] = jp.where(done | (state.info['step'] > 500), 0, state.info['step'])

        state.metrics['total_dist'] = brax.math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jp.float32(done)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State, state_info: dict, obs_history: Array) -> Array:
        inv_torso_rot = brax.math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = brax.math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate([
            jp.array([local_rpyrate[2]]) * 0.25,                # yaw rate
            brax.math.rotate(jp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
            state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
            pipeline_state.q[7:] - self._default_pose,           # motor angles
            state_info['last_act'],                              # last action
        ])
        # clip + noise
        obs = jp.clip(obs, -100, 100) + self._obs_noise * jax.random.uniform(
            state_info['rng'], obs.shape, minval=-1, maxval=1
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)
        return obs

    # Reward functions
    def _reward_lin_vel_z(self, xd: Motion) -> Array:
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> Array:
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> Array:
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = brax.math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: Array) -> Array:
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: Array, last_act: Array) -> Array:
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(self, cmd: Array, x: Transform, xd: Motion) -> Array:
        local_vel = brax.math.rotate(xd.vel[0], brax.math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(cmd[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self, cmd: Array, x: Transform, xd: Motion) -> Array:
        base_ang_vel = brax.math.rotate(xd.ang[0], brax.math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(cmd[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(self, air_time: Array, first_contact: Array, cmd: Array) -> Array:
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (brax.math.normalize(cmd[:2])[1] > 0.05)
        return rew_air_time

    def _reward_stand_still(self, cmd: Array, joint_angles: Array) -> Array:
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (brax.math.normalize(cmd[:2])[1] < 0.1)

    def _reward_foot_slip(self, pipeline_state: base.State, contact_filt: Array) -> Array:
        pos = pipeline_state.site_xpos[self._feet_site_id]
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: Array, step: Array) -> Array:
        return done & (step < 500)

# Register environment with brax
envs.register_environment('barkour', BarkourEnv)

# -------------------------------------------------------------------
# 3. Training + Data Collection
# -------------------------------------------------------------------
def train_and_collect_data(
    base_dataset_path: str = './dataset',
    run_idx: int = 0,
    num_train_steps: int = 100_000,
    num_episodes: int = 1000,
    max_episode_steps: int = 1000,
):
    """
    1) Creates a random domain config, 
    2) Trains PPO, 
    3) Collects episodes with Return-To-Go, 
    4) Saves config.json + dataset.pt.
    """

    # 1) Generate random config & create run folder
    config = QuadWorldConfig.generate_random_config()
    
    folder_id = secrets.token_hex(3)  # 6 hex digits
    save_folder = Path(base_dataset_path) / folder_id
    save_folder.mkdir(parents=True, exist_ok=True)

    config_json_path = save_folder / 'config.json'
    with open(config_json_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # 2) Create environment with domain randomization
    env_name = 'barkour'
    train_env = envs.get_environment(env_name, config=config, seed=run_idx)
    eval_env = envs.get_environment(env_name, config=config, seed=run_idx + 1234)

    # 3) Prepare the PPO training
    ckpt_path = epath.Path(f'/tmp/quad_joystick_run_{run_idx}/ckpts')
    ckpt_path.mkdir(parents=True, exist_ok=True)

    def policy_params_fn(current_step, make_policy, params):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f'{current_step}'
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    make_networks_factory = functools.partial(
        make_ppo_networks, 
        policy_hidden_layer_sizes=(128, 128, 128, 128)
    )

    # TQDM for training progress
    reward_log = []
    reward_std_log = []
    episode_length_log = []
    total_loss_log = []
    entropy_loss_log = []
    pbar = tqdm(total=num_train_steps, desc=f"[Run {run_idx}] Training PPO", unit="steps")
    
    def progress_fn(num_steps, metrics):
        pbar.n = num_steps
        numeric_metrics = {k: float(v) for k, v in metrics.items() if True}
        # Append each metric if it exists
        if 'eval/episode_reward' in numeric_metrics:
            reward_log.append((num_steps, numeric_metrics['eval/episode_reward']))
        if 'eval/episode_reward_std' in numeric_metrics:
            reward_std_log.append((num_steps, numeric_metrics['eval/episode_reward_std']))
        if 'eval/avg_episode_length' in numeric_metrics:
            episode_length_log.append((num_steps, numeric_metrics['eval/avg_episode_length']))
        if 'training/total_loss' in numeric_metrics:
            total_loss_log.append((num_steps, numeric_metrics['training/total_loss']))
        if 'training/entropy_loss' in numeric_metrics:
            entropy_loss_log.append((num_steps, numeric_metrics['training/entropy_loss']))

        # Display some of them in the progress bar
        pbar.set_postfix({
            'reward': numeric_metrics.get('eval/episode_reward', 0),
            'len': numeric_metrics.get('eval/avg_episode_length', 0),
            'loss': numeric_metrics.get('training/total_loss', 0),
            'entropy': numeric_metrics.get('training/entropy_loss', 0)
        })
        pbar.refresh()


    train_fn = functools.partial(
        ppo_train, ## Best among brax PPO implementations
        num_timesteps=num_train_steps, ## Total time steps for training
        num_evals=10, ## Logging frequency and evaluation frequency
        reward_scaling=1,
        episode_length=max_episode_steps,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.98, ## Seems to give the best results
        learning_rate=3.0e-4,
        entropy_cost=1e-2,
        num_envs=1024,
        batch_size=256,
        network_factory=make_networks_factory,
        policy_params_fn=policy_params_fn,
        seed=run_idx,
    )

    # 4) Train
    make_inference_fn, params, _ = train_fn(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=progress_fn
    )
    pbar.close()

    # Build jit-ed policy
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    # 5) Collect episodes
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    all_episodes = []

    config_vector = jp.array(config.config_to_vector()).copy()
    rng_master = jax.random.PRNGKey(run_idx + 999)

    print(f"[RUN: {run_idx}] Starting data collection. Saving to {save_folder}")
    for ep_i in tqdm(range(num_episodes), desc="[Data Collection]"):
        rng_master, rng_reset = jax.random.split(rng_master)
        state = jit_reset(rng_reset)

        raw_episode_data = []
        done_flag = 0
        steps = 0

        while not done_flag and steps < max_episode_steps:
            rng_master, rng_action = jax.random.split(rng_master)
            action, _ = jit_inference_fn(state.obs, rng_action)

            next_state = jit_step(state, action)
            raw_episode_data.append(
                (action, state.obs, float(state.done), float(state.reward))
            )

            state = next_state
            done_flag = float(state.done)
            steps += 1

        # 6) Return-to-Go
        running_sum = 0.0
        for i in reversed(range(len(raw_episode_data))):
            running_sum += raw_episode_data[i][3]
            a_i, o_i, d_i, _imm = raw_episode_data[i]
            raw_episode_data[i] = (a_i, o_i, d_i, running_sum)

        # 7) Convert to Torch
        episode_actions = []
        episode_states = []
        episode_rewards = []
        episode_dones = []
        episode_configs = []

        for (a, o, d, r2g) in raw_episode_data:
            a_np = np.array(a).copy()
            o_np = np.array(o).copy()
            c_np = np.array(config_vector).copy()

            episode_actions.append(torch.tensor(a_np, dtype=torch.float32))
            episode_states.append(torch.tensor(o_np, dtype=torch.float32))
            episode_dones.append(torch.tensor(d, dtype=torch.float32))
            episode_rewards.append(torch.tensor(r2g, dtype=torch.float32))
            episode_configs.append(torch.tensor(c_np, dtype=torch.float32))

        if len(episode_actions) == 0:
            continue  # skip empty episodes

        episode_actions = torch.stack(episode_actions)
        episode_states = torch.stack(episode_states)
        episode_rewards = torch.stack(episode_rewards)
        episode_dones = torch.stack(episode_dones)
        episode_configs = torch.stack(episode_configs)

        episode_dict = {
            "actions": episode_actions,
            "states": episode_states,
            "rewards": episode_rewards,
            "dones": episode_dones,
            "config": episode_configs
        }
        all_episodes.append(episode_dict)

    # 8) Save dataset
    dataset_path = save_folder / "dataset.pt"
    torch.save(all_episodes, dataset_path)

    ## Save the reward plot
    plot_training_metrics(
    logs_dict={
        "Eval Reward": reward_log,
        "Reward Std": reward_std_log,
        "Episode Length": episode_length_log,
        "Total Loss": total_loss_log,
        "Entropy Loss": entropy_loss_log
    },
        save_path=save_folder / 'learning_curve.png'
    )

    # JIT versions
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # Save mp4 for a Sample trajectory
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for _ in range(300):  # ~6 seconds
        rng, subkey = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, subkey)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    # Render + Save
    video_frames = eval_env.render(rollout, camera='track', width=320, height=240)
    media.write_video(save_folder / "sample_episode.mp4", video_frames, fps=int(1/eval_env.dt))

    print(f"[RUN: {run_idx}] Done. Saved dataset:\n  {dataset_path}")
    print(f"[RUN: {run_idx}] Config JSON:\n  {config_json_path}")

    return all_episodes

# -------------------------------------------------------------------
# Example usage if run as a script:
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Running a single train_and_collect_data call ...")

    for i in range(20):
        # Quick test call:
        # Adjust num_train_steps, etc. as desired.
        episodes = train_and_collect_data(
            base_dataset_path="./dataset", 
            run_idx=i,
            num_train_steps=100_000_000,
            num_episodes=250,      # smaller for quick test
            max_episode_steps=1000
        )
    print("Done. Exiting script.")
