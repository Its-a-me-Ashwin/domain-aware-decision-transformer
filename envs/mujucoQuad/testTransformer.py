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
        state_dim=465,  # e.g. dimension of state vectors in this case it is discreate so we use one hot encoding
        act_dim=12,     # e.g. dimension of action vectors same as above
        config_dim=7,  # e.g. dimension of config vectors
        hidden_size=1500,  # must be multiple of 12 for typical transformer implementations
        n_ctx=ctx_size,     # context size if your model is built for sequence data
        n_positions=ctx_size, 
        max_ep_len = ctx_size//4,
        action_tanh=True, # Forces the use of soft max activation for descreate actions
        state_tanh=True
    )

    model.load_state_dict(torch.load("./emt_model_epoch_15.pth", map_location=device))

    config = GridWorldConfig(action_inverter=False, slip_factor=0.0, wormhole_state=-0.9)
    env = GridWorldEnv(config=config, gui=True)
    

    eval = Evaluator(env, model, 1000, device)

    eval.evaluate()