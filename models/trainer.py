import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.emt import EMT
#from collectTrajectories import RLDataLoader

## For debug shit
f = open("debug.log", "w+", encoding="utf-8")
printed = True

class Hyperparameters:
    def __init__(
        self,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        epochs=100,
        betas=(0.9, 0.999),
        warmup_steps=1000,
        actionType="continious",
        stateType="continious",
        max_grad_norm=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.actionType = actionType
        self.stateType = stateType


class Trainer:
    def __init__(self, model, data_loader, hyperparams):
        """
        Args:
            model: Your EMT model instance
            data_loader: A PyTorch DataLoader object that yields batches of data
            hyperparams: An instance of Hyperparameters
        """
        self.model = model.to(hyperparams.device)
        self.data_loader = data_loader  # This is a DataLoader, not the RLDataLoader dataset
        self.hyperparams = hyperparams
        
        self.loss_fn = nn.MSELoss()
        if self.hyperparams.actionType == "continious":
            self.loss_act_fn = nn.MSELoss()
        else:
            print("Using CROSS ENTROPY FOR ACTION")
            self.loss_act_fn = nn.CrossEntropyLoss()

        if self.hyperparams.stateType == "continious":
            self.loss_state_fn = nn.MSELoss()
        else:
            print("Using CROSS ENTROPY FOR STATES")
            self.loss_state_fn = nn.CrossEntropyLoss()


        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=hyperparams.lr,
            weight_decay=hyperparams.weight_decay,
            betas=hyperparams.betas
        )

        # Learning Rate Scheduler (Cosine with Warmup)
        # (Below is a simple CosineAnnealingLR; you can combine with warmup if desired.)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=hyperparams.epochs
        )

        # Mixed Precision (optional)
        self.scaler = torch.cuda.amp.GradScaler() if "cuda" in hyperparams.device else None
        print("Scaler:", self.scaler, file=f)

    def train(self):
        global printed
        self.model.train()
        device = self.hyperparams.device

        # -------------------------------------------
        # 0) Initialize running averages outside the epoch loop
        # -------------------------------------------
        running_act_loss = 1.0
        running_state_loss = 1.0
        running_return_loss = 1.0
        alpha = 0.99  # smoothing factor for exponential moving average

        for epoch in range(self.hyperparams.epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{self.hyperparams.epochs}", leave=False)

            for batch in progress_bar:
                # ---------------------------
                # 1) Prepare your batch
                # ---------------------------
                states = batch["state"].to(device)   # [B, state_dim]
                actions = batch["action"].to(device) # [B] or [B, action_dim]
                rewards = batch["reward"].to(device) # [B]
                configs = batch["config"].to(device) # [B, config_dim]

                if actions.dim() == 1:
                    # Ensure we always have shape [B, action_dim]
                    actions = actions.unsqueeze(-1)

                # Single-step "returns_to_go" = reward
                returns_to_go = rewards

                # Create timesteps: [B], e.g. range(0..B-1)
                timesteps = torch.arange(states.shape[0], device=device, dtype=torch.long)

                # Reshape to seq_len=1
                states = states.unsqueeze(1)    # [B, 1, state_dim]
                actions = actions.unsqueeze(1)  # [B, 1, action_dim]
                configs = configs.unsqueeze(1)  # [B, 1, config_dim]

                # Convert to long if discrete (for embedding + CE)
                if self.hyperparams.stateType != "continious":
                    states = states.to(torch.long)
                if self.hyperparams.actionType != "continious":
                    actions = actions.to(torch.long)

                # [B, 1, 1] for returns_to_go
                returns_to_go = returns_to_go.unsqueeze(1).unsqueeze(-1)
                # [B, 1] for timesteps
                timesteps = timesteps.unsqueeze(1)

                # Clear gradients
                self.optimizer.zero_grad()

                if self.scaler is not None:
                    # ---------------------------
                    # 2) Mixed precision forward pass
                    # ---------------------------
                    with torch.cuda.amp.autocast():
                        state_preds, action_preds, return_preds = self.model(
                            states,
                            actions,
                            rewards=None,
                            returns_to_go=returns_to_go,
                            timesteps=timesteps,
                            config=configs
                        )

                        # ---------------------------
                        # 3) Compute partial losses
                        # ---------------------------
                        # Action Loss (CE or MSE)
                        if self.hyperparams.actionType != "continious":
                            # Cross Entropy: convert [B, 1, act_dim] from one-hot → class idx
                            actions_idx = actions.argmax(dim=-1)  # [B,1]
                            act_loss = self.loss_act_fn(
                                action_preds.view(-1, self.model.act_dim),  # [B*1, act_dim]
                                actions_idx.view(-1)                       # [B*1]
                            )
                        else:
                            act_loss = self.loss_act_fn(action_preds, actions.float())

                        # State Loss (CE or MSE)
                        if self.hyperparams.stateType != "continious":
                            states_idx = states.argmax(dim=-1)
                            state_loss = self.loss_act_fn(
                                state_preds.view(-1, self.model.state_dim),
                                states_idx.view(-1)
                            )
                        else:
                            state_loss = self.loss_state_fn(state_preds, states.float())

                        # Reward Loss (usually MSE)
                        reward_loss = self.loss_fn(return_preds, returns_to_go)

                        # Make each a scalar
                        act_loss_ = act_loss.mean()
                        state_loss_ = state_loss.mean()
                        reward_loss_ = reward_loss.mean()

                    # ---------------------------
                    # 4) Update running averages (no grad)
                    # ---------------------------
                    with torch.no_grad():
                        running_act_loss = alpha * running_act_loss + (1 - alpha) * act_loss_.item()
                        running_state_loss = alpha * running_state_loss + (1 - alpha) * state_loss_.item()
                        running_return_loss = alpha * running_return_loss + (1 - alpha) * reward_loss_.item()

                    # ---------------------------
                    # 5) Normalize partial losses by their running averages
                    # ---------------------------
                    act_loss_norm = act_loss_ / (running_act_loss + 1e-8)
                    state_loss_norm = state_loss_ / (running_state_loss + 1e-8)
                    return_loss_norm = reward_loss_ / (running_return_loss + 1e-8)

                    # Final combined loss
                    final_loss = act_loss_norm + state_loss_norm + return_loss_norm

                    # Backprop with AMP
                    self.scaler.scale(final_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    loss_val = final_loss.item()

                else:
                    # ---------------------------
                    # 2) No mixed precision
                    # ---------------------------
                    state_preds, action_preds, return_preds = self.model(
                        states,
                        actions,
                        rewards=None,
                        returns_to_go=returns_to_go,
                        timesteps=timesteps,
                        config=configs
                    )

                    # Action Loss
                    if self.hyperparams.actionType != "continious":
                        actions_idx = actions.argmax(dim=-1)
                        act_loss = self.loss_act_fn(
                            action_preds.view(-1, self.model.act_dim),
                            actions_idx.view(-1)
                        )
                    else:
                        act_loss = self.loss_act_fn(action_preds, actions.float())

                    # State Loss
                    if self.hyperparams.stateType != "continious":
                        states_idx = states.argmax(dim=-1)
                        state_loss = self.loss_act_fn(
                            state_preds.view(-1, self.model.state_dim),
                            states_idx.view(-1)
                        )
                    else:
                        state_loss = self.loss_state_fn(state_preds, states.float())

                    # Reward Loss
                    reward_loss = self.loss_fn(return_preds, returns_to_go)

                    act_loss_ = act_loss.mean()
                    state_loss_ = state_loss.mean()
                    reward_loss_ = reward_loss.mean()

                    # ---------------------------
                    # 4) Update running averages
                    # ---------------------------
                    with torch.no_grad():
                        running_act_loss = alpha * running_act_loss + (1 - alpha) * act_loss_.item()
                        running_state_loss = alpha * running_state_loss + (1 - alpha) * state_loss_.item()
                        running_return_loss = alpha * running_return_loss + (1 - alpha) * reward_loss_.item()

                    # ---------------------------
                    # 5) Normalize partial losses
                    # ---------------------------
                    act_loss_norm = act_loss_ / (running_act_loss + 1e-8)
                    state_loss_norm = state_loss_ / (running_state_loss + 1e-8)
                    return_loss_norm = reward_loss_ / (running_return_loss + 1e-8)

                    final_loss = act_loss_norm + state_loss_norm + return_loss_norm

                    # Backprop
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams.max_grad_norm)
                    self.optimizer.step()

                    loss_val = final_loss.item()

                # Step the scheduler
                self.scheduler.step()

                epoch_loss += loss_val
                progress_bar.set_postfix(loss=f"{loss_val:.6f}")

            print(f"Epoch {epoch+1}/{self.hyperparams.epochs}, Loss: {epoch_loss / len(self.data_loader):.6f}")



    def save_model(self, path="emt_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="emt_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.hyperparams.device))
        self.model.to(self.hyperparams.device)
        print(f"Model loaded from {path}")
        return self.model


if __name__ == "__main__":
    # -----------------------------
    # 1) Setup Hyperparameters
    # -----------------------------
    hyperparams = Hyperparameters(
        batch_size=256, 
        lr=1e-4, 
        weight_decay=1e-4, 
        epochs=4
    )

    # -----------------------------
    # 2) Load Dataset & Create DataLoader
    # -----------------------------
    # RLDataLoader is a Dataset that concatenates data from ./dataset subdirectories
    dataset = RLDataLoader(dataset_root="./dataset")

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
        state_dim=4,  # e.g. dimension of state vectors
        act_dim=2,     # e.g. dimension of action vectors
        config_dim=9,  # e.g. dimension of config vectors
        hidden_size=600,  # must be multiple of 12 for typical transformer implementations
        n_ctx=1_000     # context size if your model is built for sequence data
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
