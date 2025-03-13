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
from models.trainer import Hyperparameters, Trainer
from models.dataloader import RLDataLoader


if __name__ == "__main__":
    # -----------------------------
    # 1) Setup Hyperparameters
    # -----------------------------
    hyperparams = Hyperparameters(
        batch_size=512, 
        lr=1e-4, 
        weight_decay=1e-4,
        actionType="discreate", 
        stateType="discreate",
        epochs=10
    )

    # -----------------------------
    # 2) Load Dataset & Create DataLoader
    # -----------------------------
    # RLDataLoader is a Dataset that concatenates data from ./dataset subdirectories
    dataset = RLDataLoader(dataset_root="./dataset", action_cat=True, states_cat=True)

    # Now wrap it into a PyTorch DataLoader for batching
    train_data_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        drop_last=True  # optional, to keep batch sizes consistent
    )

    # -----------------------------
    # 3) Initialize Your Model
    # -----------------------------
    # Make sure to adjust these dimensions based on your actual data
    model = EMT(
        state_dim=10*10,  # e.g. dimension of state vectors in this case it is discreate so we use one hot encoding
        act_dim=4,     # e.g. dimension of action vectors same as above
        config_dim=4,  # e.g. dimension of config vectors
        hidden_size=600,  # must be multiple of 12 for typical transformer implementations
        n_ctx=1_000,     # context size if your model is built for sequence data
        max_ep_len = 1_000,
        action_tanh=False, # Forces the use of soft max activation for descreate actions
        state_tanh=False
    )

    # -----------------------------
    # 4) Train the Model
    # -----------------------------
    trainer = Trainer(model, train_data_loader, hyperparams)
    trainer.train()

    # -----------------------------
    # 5) Save the Trained Model
    # -----------------------------
    trainer.save_model("emt_model.pth")