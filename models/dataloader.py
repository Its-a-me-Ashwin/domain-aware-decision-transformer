import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RLDataLoader(Dataset):
    """
    PyTorch Dataset that loads all data from each domain subdirectory
    in ./dataset. Each subdirectory should contain:
        - dataset.pt  (the dictionary with 'action', 'state', 'done', 'reward', 'config')
        - config.json (the environment parameters)
    This loader concatenates all data points from all subdirectories 
    and makes them available for training (optionally with shuffle in a DataLoader).
    """

    def __init__(self, dataset_root="./dataset", action_cat=False, states_cat=False):
        super().__init__()
        self.dataset_root = dataset_root
        self.all_actions = []
        self.all_states = []
        self.all_dones = []
        self.all_rewards = []
        self.all_configs = []

        self.action_cat = action_cat
        self.state_cat = states_cat
        # Iterate over all subdirectories in dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset root '{self.dataset_root}' does not exist.")

        for subdir in os.listdir(self.dataset_root):
            config_dir = os.path.join(self.dataset_root, subdir)
            if not os.path.isdir(config_dir):
                # Skip files that are not directories
                continue

            dataset_pt = os.path.join(config_dir, "dataset.pt")
            if not os.path.isfile(dataset_pt):
                # Skip directories that do not contain dataset.pt
                continue

            # Load the .pt file
            loaded_data = torch.load(dataset_pt)
            # Expecting keys: 'action', 'state', 'done', 'reward', 'config'
            actions = loaded_data["action"]
            states = loaded_data["state"]
            dones = loaded_data["done"]
            rewards = loaded_data["reward"]
            configs = loaded_data["config"]  # repeated for each data point

            # Concatenate these data with the global lists
            if self.action_cat:
                self.all_actions.append(actions.to(torch.int16))  # Convert actions to int16
            else:
                self.all_states.append(states)

            if self.state_cat:
                self.all_states.append(states.to(torch.int16))  # Convert states to int16
            else:
                self.all_states.append(states)
            self.all_dones.append(dones)
            self.all_rewards.append(rewards)
            self.all_configs.append(configs)

        # Stack everything into big tensors
        if len(self.all_actions) == 0:
            raise ValueError("No data found in any subdirectory under dataset_root.")

        self.all_actions = torch.cat(self.all_actions, dim=0)
        self.all_states = torch.cat(self.all_states, dim=0)
        self.all_dones = torch.cat(self.all_dones, dim=0)
        self.all_rewards = torch.cat(self.all_rewards, dim=0)
        self.all_configs = torch.cat(self.all_configs, dim=0)

        print("Dataset", self.all_actions.dtype)

        #print("In loader", self.all_actions.size(), self.all_states.size(), self.all_rewards.size())
        # Optionally, you could randomize the order right here, but typically
        # you'd pass `shuffle=True` to the DataLoader in your training script.
        # If you want to shuffle in-memory right now, you can do:
        # permutation = torch.randperm(self.all_actions.size(0))
        # self.all_actions = self.all_actions[permutation]
        # self.all_states = self.all_states[permutation]
        # self.all_dones = self.all_dones[permutation]
        # self.all_rewards = self.all_rewards[permutation]
        # self.all_configs = self.all_configs[permutation]

    def __len__(self):
        return self.all_actions.shape[0]

    def __getitem__(self, idx):
        return {
            "action": self.all_actions[idx],
            "state": self.all_states[idx],
            "done": self.all_dones[idx],
            "reward": self.all_rewards[idx],
            "config": self.all_configs[idx],
        }