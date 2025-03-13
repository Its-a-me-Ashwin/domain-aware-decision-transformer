import os
import json
import copy
import string
import random
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gym import Wrapper
from env import GridWorldEnv, GridWorldConfig


class DomainRandomizationWrapper(Wrapper):
    """
    Wrapper for optionally applying domain randomization to an environment.
    """
    def __init__(self, env, randomize=False):
        super().__init__(env)
        self.randomize = randomize

    def reset(self, **kwargs):
        if self.randomize:
            pass  # Implement randomization logic if needed
        return super().reset(**kwargs)


def generate_random_hash(num_chars=10):
    """Generate a random string hash (letters only)."""
    return ''.join(random.choices(string.ascii_letters, k=num_chars))


def train_rl_agent_and_collect_data(
    env_config,
    num_episodes=5000,
    reward_threshold=25.0,
    max_training_steps=200000,
    n_envs=8
):
    """
    Trains an RL agent (PPO) on the given environment configuration.
    Collects (action, state, done, reward, config) data,
    where action and state are saved as one-hot (discrete) tensors.
    """

    dataset_root = os.path.abspath("./dataset")
    os.makedirs(dataset_root, exist_ok=True)

    # Vectorized environment for stable training
    vec_env = make_vec_env(
        lambda: DomainRandomizationWrapper(
            GridWorldEnv(config=env_config),
            randomize=False
        ),
        n_envs=n_envs,
        seed=np.random.randint(2**31 - 1),
        vec_env_cls=DummyVecEnv,
    )

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        batch_size=512,
        gamma=0.99,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "auto",
    )

    # Evaluation environment
    eval_env = DomainRandomizationWrapper(
        GridWorldEnv(config=env_config),
        randomize=False
    )

    stop_training_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_training_callback,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=max_training_steps,
        callback=eval_callback,
        progress_bar=True
    )

    if eval_callback.best_mean_reward < reward_threshold:
        print("Best mean reward did not reach the threshold. Skipping data collection.")
        return

    config_vec = GridWorldConfig.config_to_vector(env_config)
    config_tensor = torch.tensor(config_vec, dtype=torch.float32)

    # Save trained model
    config_hash = generate_random_hash()
    config_directory = os.path.join(dataset_root, config_hash)
    os.makedirs(config_directory, exist_ok=True)
    model_path = os.path.join(config_directory, "ppo_gridworld_agent")
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Collect data using trained agent
    data_collection_config = copy.deepcopy(env_config)
    data_collection_config.visual = False  # Disable visualization during data collection
    model = PPO.load(model_path, device="cuda" if torch.cuda.is_available() else "auto")

    rollout_data = []
    for _ in range(num_episodes):
        collect_env = GridWorldEnv(config=data_collection_config)
        obs, _ = collect_env.reset()
        done = False
        episode_data = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # ensure it's an int
            try:
                next_obs, reward, done, _, _ = collect_env.step(action)
            except Exception as e:
                print("Failed to collect data", e, action)
                break
            episode_data.append([action, next_obs, done, reward])
            obs = next_obs
        collect_env.close()

        # Compute return-to-go rewards
        running_sum = 0.0
        for i in reversed(range(len(episode_data))):
            running_sum += episode_data[i][3]
            episode_data[i][3] = running_sum

        rollout_data.extend(episode_data)

    # Convert actions and states to one-hot (discrete) representations
    one_hot_actions = []
    one_hot_states = []
    done_tensors = []
    reward_tensors = []

    for (action, state, done_flag, reward_val) in rollout_data:
        # One-hot encode action (4 possible actions: 0,1,2,3)
        action_tensor = F.one_hot(
            torch.tensor(action, dtype=torch.long), 
            num_classes=4
        ).to(torch.int)  # or torch.float, but int is valid for discrete data

        # State should be a 1D of size 2 (x, y) each in [0..9]
        x = int(state[0])
        y = int(state[1])
        idx = 10 * y + x
        state_tensor = F.one_hot(
            torch.tensor(idx, dtype=torch.long),
            num_classes=100
        ).to(torch.int)

        one_hot_actions.append(action_tensor)
        one_hot_states.append(state_tensor)
        done_tensors.append(torch.tensor(done_flag, dtype=torch.bool))
        reward_tensors.append(torch.tensor(reward_val, dtype=torch.float32))

    N = len(one_hot_actions)
    config_tensor_expanded = config_tensor.unsqueeze(0).expand(N, -1)

    # Save the dataset
    dataset_path = os.path.join(config_directory, "dataset.pt")
    torch.save({
        "action": torch.stack(one_hot_actions),
        "state": torch.stack(one_hot_states),
        "done": torch.stack(done_tensors),
        "reward": torch.stack(reward_tensors),
        "config": config_tensor_expanded
    }, dataset_path)

    print(f"Dataset saved to: {dataset_path}")

    # Save config JSON
    config_json_path = os.path.join(config_directory, "config.json")
    with open(config_json_path, "w") as f:
        json.dump(env_config.to_dict(), f, indent=4)
    print(f"Config JSON saved to: {config_json_path}")


if __name__ == "__main__":
    num_configs = 100

    for i in range(num_configs):
        random_config = GridWorldConfig.generate_random_config()

        # Allow manual testing of environment
        random_config.visual = True
        env = GridWorldEnv(config=random_config)
        # env.play_keyboard()  # For manual play if desired
        random_config.visual = False

        train_rl_agent_and_collect_data(random_config)
