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


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    keys = ["states", "actions", "rewards", "config", "timesteps", "dones"]
    padded = {}
    lengths = [b["states"].shape[0] for b in batch]
    max_len = max(lengths)

    for key in keys:
        seqs = [b[key] for b in batch]
        if seqs[0].dim() == 1:
            padded[key] = pad_sequence(seqs, batch_first=True)  # [B, T]
        else:
            padded[key] = pad_sequence(seqs, batch_first=True)  # [B, T, D]

    # Generate attention mask: 1 for real token, 0 for padding
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    padded["attention_mask"] = attention_mask  # [B, T]

    return padded


if __name__ == "__main__":
    # -----------------------------
    # 1) Setup Hyperparameters
    # -----------------------------
    hyperparams = Hyperparameters(
        batch_size=2, # 256 + 128 
        lr=1e-4, 
        weight_decay=1e-4,
        actionType="continious", 
        stateType="continious",
        epochs=50
    )

    max_seq_len = 100
    ctx_size = max_seq_len * 4 # Context size need 8 4 due to all the modalities

    # -----------------------------
    # 2) Load Dataset & Create DataLoader
    # -----------------------------
    # RLDataLoader is a Dataset that concatenates data from ./dataset subdirectories
    dataset = RLDataLoader(dataset_root="./dataset", action_cat=False, state_cat=False, max_seq_len=max_seq_len, percentage=0.25)

    # Now wrap it into a PyTorch DataLoader for batching
    train_data_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        drop_last=True,  # optional, to keep batch sizes consistent
        collate_fn=collate_fn  # use custom collate function for padding
    )

    # -----------------------------
    # 3) Initialize Your Model
    # -----------------------------
    # Make sure to adjust these dimensions based on your actual data
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

    # -----------------------------
    # 4) Train the Model
    # -----------------------------
    trainer = Trainer(model, train_data_loader, hyperparams)
    trainer.train()

    # -----------------------------
    # 5) Save the Trained Model
    # -----------------------------
    trainer.save_model("emt_model.pth")