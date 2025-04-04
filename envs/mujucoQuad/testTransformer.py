import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Get the absolute path of the project root (assuming `envs` is inside it)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Now import the module
from models.emt import EMT
from models.evaluator import Evaluator
from env import GridWorldEnv, GridWorldConfig  

device = "cpu"

if __name__ == "__main__":
    model = EMT(
        state_dim=2,  # e.g. dimension of state vectors
        act_dim=1,     # e.g. dimension of action vectors
        config_dim=4,  # e.g. dimension of config vectors
        hidden_size=600,  # must be multiple of 12 for typical transformer implementations
        n_ctx=1_000,     # context size if your model is built for sequence data
        action_tanh=False, # Forces the use of soft max activation for descreate actions
        state_tanh=False
    )

    model.load_state_dict(torch.load("./emt_model.pth", map_location=device))

    config = GridWorldConfig(action_inverter=False, slip_factor=0.0, wormhole_state=-0.9)
    env = GridWorldEnv(config=config, gui=True)
    

    eval = Evaluator(env, model, 1000, device)

    eval.evaluate()