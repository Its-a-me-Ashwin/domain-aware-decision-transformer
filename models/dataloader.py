import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class RLDataLoader(Dataset):
    def __init__(self, dataset_root, action_cat=False, state_cat=False, max_seq_len=None, percentage=1.0):
        super().__init__()
        self.dataset_root = dataset_root
        self.action_cat = action_cat
        self.state_cat = state_cat
        self.max_seq_len = max_seq_len
        self.percentage = percentage

        self.normalized_episodes = []

        # Temporary storage to collect for normalization
        all_states = []
        all_actions = []
        all_rewards = []
        all_raw_episodes = []

        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist.")

        for subdir in os.listdir(dataset_root):
            subdir_path = os.path.join(dataset_root, subdir)
            if not os.path.isdir(subdir_path):
                continue

            dataset_file = os.path.join(subdir_path, "dataset.pt")
            config_file = os.path.join(subdir_path, "config.json")

            if not os.path.exists(dataset_file):
                print(f"Skipping {subdir} (missing dataset.pt)")
                continue

            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Missing config.json in '{subdir_path}'")

            with open(config_file, "r") as cf:
                config_dict = json.load(cf)

            config_array = np.array(list(config_dict.values()), dtype=np.float32)
            config_tensor = torch.from_numpy(config_array)

            episodes = torch.load(dataset_file)

            for ep in episodes:
                states = ep["states"].clone().detach().float()
                actions = ep["actions"].clone().detach().float()
                rewards = ep["rewards"].clone().detach().float()
                dones = ep["dones"].clone().detach().bool()

                if self.state_cat:
                    states = states.to(torch.int16)
                if self.action_cat:
                    actions = actions.to(torch.int16)

                all_raw_episodes.append({
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                    "config": config_tensor
                })

                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)

        # Shuffle and apply percentage filtering
        total_eps = len(all_raw_episodes)
        np.random.shuffle(all_raw_episodes)
        selected_eps = int(total_eps * self.percentage)
        all_raw_episodes = all_raw_episodes[:selected_eps]

        # Global normalization
        all_states_tensor = torch.cat(all_states, dim=0)
        all_actions_tensor = torch.cat(all_actions, dim=0)
        all_rewards_tensor = torch.cat(all_rewards, dim=0)

        self.state_mean = all_states_tensor.mean(0, keepdim=True)
        self.state_std = all_states_tensor.std(0, keepdim=True) + 1e-6
        self.action_mean = all_actions_tensor.mean(0, keepdim=True)
        self.action_std = all_actions_tensor.std(0, keepdim=True) + 1e-6
        self.reward_mean = all_rewards_tensor.mean()
        self.reward_std = all_rewards_tensor.std() + 1e-6

        # Normalize and construct final episodes
        for ep in all_raw_episodes:
            states = (ep["states"] - self.state_mean) / self.state_std
            actions = (ep["actions"] - self.action_mean) / self.action_std
            rewards = (ep["rewards"] - self.reward_mean) / self.reward_std
            dones = ep["dones"]

            T = states.shape[0]
            config = ep["config"].unsqueeze(0).repeat(T, 1)

            if self.max_seq_len is not None:
                states = states[:self.max_seq_len]
                actions = actions[:self.max_seq_len]
                rewards = rewards[:self.max_seq_len]
                dones = dones[:self.max_seq_len]
                config = config[:self.max_seq_len]

            timesteps = torch.arange(states.shape[0], dtype=torch.long)

            self.normalized_episodes.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "config": config,
                "timesteps": timesteps
            })

        print(f"Loaded {len(self.normalized_episodes)} / {total_eps} episodes ({self.percentage * 100:.1f}%) from '{dataset_root}'")
        print(f"State mean/std: {self.state_mean.shape}, {self.state_std.shape}")
        print(f"Action mean/std: {self.action_mean.shape}, {self.action_std.shape}")

    def __len__(self):
        return len(self.normalized_episodes)

    def __getitem__(self, idx):
        return self.normalized_episodes[idx]

    def unnormalize_state(self, norm_state):
        return norm_state * self.state_std + self.state_mean

    def unnormalize_action(self, norm_action):
        return norm_action * self.action_std + self.action_mean

    def unnormalize_reward(self, norm_reward):
        return norm_reward * self.reward_std + self.reward_mean
