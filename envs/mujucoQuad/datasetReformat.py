import os
import torch
from tqdm import tqdm

def flatten_dataset(dataset_path='./dataset'):
    for subdir in tqdm(os.listdir(dataset_path), desc="Processing dataset folders"):
        config_dir = os.path.join(dataset_path, subdir)
        dataset_pt = os.path.join(config_dir, "dataset.pt")

        if not os.path.isdir(config_dir) or not os.path.isfile(dataset_pt):
            continue  # Skip invalid dirs

        # Load list of episode dictionaries
        episodes = torch.load(dataset_pt)
        
        if isinstance(episodes, dict):
            print(f"Already flattened: {subdir}, skipping.")
            continue

        if not isinstance(episodes, list):
            print(f"Skipping {subdir}: not a list of episodes")
            continue

        # Flatten across episodes
        all_actions = []
        all_states = []
        all_rewards = []
        all_dones = []
        all_configs = []

        for ep in episodes:
            key_actions = 'actions' if 'actions' in ep else 'action'
            key_states = 'states' if 'states' in ep else 'state'
            key_dones = 'dones' if 'dones' in ep else 'done'
            key_rewards = 'rewards' if 'rewards' in ep else 'reward'

            all_actions.append(ep[key_actions])
            all_states.append(ep[key_states])
            all_rewards.append(ep[key_rewards])
            all_dones.append(ep[key_dones])
            all_configs.append(ep["config"])

        # Concatenate along time dimension
        flat_data = {
            'actions': torch.cat(all_actions, dim=0),
            'states': torch.cat(all_states, dim=0),
            'rewards': torch.cat(all_rewards, dim=0),
            'dones': torch.cat(all_dones, dim=0),
            'config': torch.cat(all_configs, dim=0),  # same config repeated
        }

        # Overwrite the file
        torch.save(flat_data, dataset_pt)
        print(f"✔ Flattened and saved: {dataset_pt}")

if __name__ == "__main__":
    flatten_dataset('./dataset')
