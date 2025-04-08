import json
import secrets
import torch
import jax
import jax.numpy as jp
import numpy as np
import functools
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp
import mediapy as media
import matplotlib.pyplot as plt

from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.ppo.networks import make_ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

# ------------------------------
# Randomization Configuration
# ------------------------------

class FrankaPushConfig:
    def __init__(
        self,
        gravity_strength: float = 9.81,
        gravity_angle_x: float = 0.0,
        gravity_angle_y: float = 0.0,
        max_motor_torque: float = 50.0,
        friction: float = 1.0,
        wind_force_x: float = 0.0,
        wind_force_y: float = 0.0,
    ):
        self.gravity_strength = gravity_strength
        self.gravity_angle_x = gravity_angle_x
        self.gravity_angle_y = gravity_angle_y
        self.max_motor_torque = max_motor_torque
        self.friction = friction
        self.wind_force_x = wind_force_x
        self.wind_force_y = wind_force_y

    @staticmethod
    def generate_random_config() -> "FrankaPushConfig":
        return FrankaPushConfig(
            gravity_strength=np.random.uniform(7.0, 15.0),
            gravity_angle_x=np.random.uniform(-0.3, 0.3),
            gravity_angle_y=np.random.uniform(-0.3, 0.3),
            max_motor_torque=np.random.uniform(20.0, 80.0),
            friction=np.random.uniform(0.1, 2.0),
            wind_force_x=np.random.uniform(-2.0, 2.0),
            wind_force_y=np.random.uniform(-2.0, 2.0),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "gravity_strength": float(self.gravity_strength),
            "gravity_angle_x": float(self.gravity_angle_x),
            "gravity_angle_y": float(self.gravity_angle_y),
            "max_motor_torque": float(self.max_motor_torque),
            "friction": float(self.friction),
            "wind_force_x": float(self.wind_force_x),
            "wind_force_y": float(self.wind_force_y),
        }

    def config_to_vector(self) -> List[float]:
        return [
            self.gravity_strength,
            self.gravity_angle_x,
            self.gravity_angle_y,
            self.max_motor_torque,
            self.friction,
            self.wind_force_x,
            self.wind_force_y,
        ]

class FrankaPushRandomizer:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_config(self) -> FrankaPushConfig:
        return FrankaPushConfig.generate_random_config()

    def apply_to_env_cfg(self, env_cfg, config: FrankaPushConfig):
        g = config.gravity_strength
        gx = config.gravity_angle_x
        gy = config.gravity_angle_y
        env_cfg.gravity = jp.array([
            g * np.sin(gx),
            g * np.sin(gy),
            -g * np.cos(gx) * np.cos(gy),
        ])
        if "robot" in env_cfg:
            env_cfg.robot.torque_limit = config.max_motor_torque
        if "block_friction" in env_cfg:
            env_cfg.block_friction = config.friction
        env_cfg.wind_force = jp.array([config.wind_force_x, config.wind_force_y, 0.0])
        return env_cfg

# -----------------------------
# Training & Data Collection
# -----------------------------

def train_and_collect_data(
    base_dataset_path: str = './dataset',
    run_idx: int = 0,
    num_train_steps: int = 100_000,
    num_episodes: int = 1000,
    max_episode_steps: int = 1000,
):
    config = FrankaPushConfig.generate_random_config()

    folder_id = secrets.token_hex(3)
    save_folder = Path(base_dataset_path) / folder_id
    save_folder.mkdir(parents=True, exist_ok=True)

    config_json_path = save_folder / 'config.json'
    with open(config_json_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    env_name = 'PandaPickCubeOrientation'
    env_cfg = registry.get_default_config(env_name)
    env_cfg.task = 'push_to_goal'

    randomizer = FrankaPushRandomizer(seed=run_idx)
    env_cfg = randomizer.apply_to_env_cfg(env_cfg, config)

    train_env = registry.load(env_name, config=env_cfg)
    eval_env = registry.load(env_name, config=env_cfg)
    # train_env = envs.get_environment(env_name, config=env_cfg, seed=run_idx)
    # eval_env = envs.get_environment(env_name, config=config, seed=run_idx + 1234)



    ckpt_path = epath.Path(f'/tmp/franka_push_run_{run_idx}/ckpts')
    ckpt_path.mkdir(parents=True, exist_ok=True)

    def policy_params_fn(current_step, make_policy, params):
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f'{current_step}'
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)

    make_networks_factory = functools.partial(make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 128, 128))

    reward_log, reward_std_log, episode_length_log, total_loss_log, entropy_loss_log = [], [], [], [], []
    pbar = tqdm(total=num_train_steps, desc=f"[Run {run_idx}] Training PPO", unit="steps")

    def progress_fn(num_steps, metrics):
        pbar.n = num_steps
        numeric_metrics = {k: float(v) for k, v in metrics.items() if True}
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
        pbar.set_postfix({
            'reward': numeric_metrics.get('eval/episode_reward', 0),
            'len': numeric_metrics.get('eval/avg_episode_length', 0),
            'loss': numeric_metrics.get('training/total_loss', 0),
            'entropy': numeric_metrics.get('training/entropy_loss', 0)
        })
        pbar.refresh()

    train_fn = functools.partial(
        ppo_train,
        num_timesteps=num_train_steps,
        num_evals=10,
        reward_scaling=1,
        episode_length=max_episode_steps,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.98,
        learning_rate=3.0e-4,
        entropy_cost=1e-2,
        num_envs=1024,
        batch_size=256,
        network_factory=make_networks_factory,
        policy_params_fn=policy_params_fn,
        seed=run_idx,
    )

    make_inference_fn, params, _ = train_fn(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=progress_fn
    )
    pbar.close()

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

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
            raw_episode_data.append((action, state.obs, float(state.done), float(state.reward)))
            state = next_state
            done_flag = float(state.done)
            steps += 1

        running_sum = 0.0
        for i in reversed(range(len(raw_episode_data))):
            running_sum += raw_episode_data[i][3]
            a_i, o_i, d_i, _imm = raw_episode_data[i]
            raw_episode_data[i] = (a_i, o_i, d_i, running_sum)

        episode_actions, episode_states, episode_rewards, episode_dones, episode_configs = [], [], [], [], []
        for (a, o, d, r2g) in raw_episode_data:
            episode_actions.append(torch.tensor(np.array(a), dtype=torch.float32))
            episode_states.append(torch.tensor(np.array(o), dtype=torch.float32))
            episode_dones.append(torch.tensor(d, dtype=torch.float32))
            episode_rewards.append(torch.tensor(r2g, dtype=torch.float32))
            episode_configs.append(torch.tensor(np.array(config_vector), dtype=torch.float32))

        if len(episode_actions) == 0:
            continue

        episode_dict = {
            "actions": torch.stack(episode_actions),
            "states": torch.stack(episode_states),
            "rewards": torch.stack(episode_rewards),
            "dones": torch.stack(episode_dones),
            "config": torch.stack(episode_configs)
        }
        all_episodes.append(episode_dict)

    dataset_path = save_folder / "dataset.pt"
    torch.save(all_episodes, dataset_path)

    video_frames = []
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for _ in range(300):
        rng, subkey = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, subkey)
        state = jit_step(state, action)
        rollout.append(state.pipeline_state)

    video_frames = eval_env.render(rollout, camera='track', width=320, height=240)
    media.write_video(save_folder / "sample_episode.mp4", video_frames, fps=int(1 / eval_env.dt))

    print(f"[RUN: {run_idx}] Done. Saved dataset:\n  {dataset_path}")
    print(f"[RUN: {run_idx}] Config JSON:\n  {config_json_path}")

    return all_episodes


if __name__ == "__main__":
    print("Running a single train_and_collect_data call ...")

    for i in range(20):
        episodes = train_and_collect_data(
            base_dataset_path="./dataset", 
            run_idx=i,
            num_train_steps=100_000_000,
            num_episodes=250,
            max_episode_steps=1000
        )
    print("Done. Exiting script.")
