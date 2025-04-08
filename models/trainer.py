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
        self.log_interval = 100


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
        self.global_step = 0
        
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
        # torch.amp.GradScaler('cuda', args...) new
        # torch.cuda.amp.GradScaler(args...) old
        self.scaler = torch.amp.GradScaler('cuda') if "cuda" in hyperparams.device else None
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

                actionKey = "actions" if "actions" in batch else "action"
                stateKey = "states" if "states" in batch else "state"
                rewardKey = "rewards" if "rewards" in batch else "reward"


                states = batch[stateKey].to(device)         # [B, T, state_dim]
                actions = batch[actionKey].to(device)       # [B, T, action_dim]
                rewards = batch[rewardKey].to(device)       # [B, T]
                configs = batch["config"].to(device)        # [B, T, config_dim]
                timesteps = batch["timesteps"].to(device)   # [B, T]
                attention_mask = batch["attention_mask"].to(device)  # [B, T]

                B, T, _ = states.shape

                # Compute returns-to-go (you can change this to discounted return if needed)
                returns_to_go = rewards.clone().unsqueeze(-1)  # [B, T, 1]

                self.optimizer.zero_grad()

                total_act_loss = 0.0
                total_state_loss = 0.0
                total_reward_loss = 0.0
                total_loss = 0.0


                for t in range(1, T):
                    # Inputs: everything before t
                    input_states = states[:, :t]            # [B, t, state_dim]
                    input_actions = actions[:, :t]          # [B, t, action_dim]
                    input_returns = returns_to_go[:, :t]    # [B, t, 1]
                    input_config = configs[:, :t]           # [B, t, config_dim]
                    input_timesteps = timesteps[:, :t]      # [B, t]

                    input_attention_mask = attention_mask[:, :t]                    # [B, t]
                    # input_attention_mask = input_attention_mask.unsqueeze(1)       # [B, 1, t]
                    # input_attention_mask = input_attention_mask.expand(-1, 4, -1)   # [B, 4, t]
                    # input_attention_mask = input_attention_mask.reshape(B, 4 * t)   # [B, 4*t]

                    target_state = states[:, t]             # [B, state_dim]
                    target_action = actions[:, t]           # [B, action_dim]
                    target_return = returns_to_go[:, t]     # [B, 1]

                    with torch.amp.autocast(device_type='cuda') if self.scaler else torch.no_grad():
                        state_preds, action_preds, return_preds = self.model(
                            input_states,
                            input_actions,
                            rewards=None,
                            returns_to_go=input_returns,
                            timesteps=input_timesteps,
                            config=input_config,
                            attention_mask=input_attention_mask
                        )


                        pred_state = state_preds[:, -1]     # Only last timestep
                        pred_action = action_preds[:, -1]
                        pred_return = return_preds[:, -1]

                        if self.hyperparams.actionType != "continious":
                            target_action_idx = target_action.argmax(dim=-1)
                            act_loss = self.loss_act_fn(pred_action.view(B, -1), target_action_idx)
                        else:
                            act_loss = self.loss_act_fn(pred_action, target_action)

                        if self.hyperparams.stateType != "continious":
                            target_state_idx = target_state.argmax(dim=-1)
                            state_loss = self.loss_state_fn(pred_state.view(B, -1), target_state_idx)
                        else:
                            state_loss = self.loss_state_fn(pred_state, target_state)

                        reward_loss = self.loss_fn(pred_return, target_return)

                        act_loss_ = act_loss.mean()
                        state_loss_ = state_loss.mean()
                        reward_loss_ = reward_loss.mean()

                        # Normalize with running averages
                        running_act_loss = alpha * running_act_loss + (1 - alpha) * act_loss_.item()
                        running_state_loss = alpha * running_state_loss + (1 - alpha) * state_loss_.item()
                        running_return_loss = alpha * running_return_loss + (1 - alpha) * reward_loss_.item()

                        act_loss_norm = act_loss_ / (running_act_loss + 1e-8)
                        state_loss_norm = state_loss_ / (running_state_loss + 1e-8)
                        return_loss_norm = reward_loss_ / (running_return_loss + 1e-8)

                        step_loss = act_loss_norm + state_loss_norm + return_loss_norm

                    if self.scaler:
                        self.scaler.scale(step_loss).backward()
                    else:
                        step_loss.backward()

                    total_act_loss += act_loss_.item()
                    total_state_loss += state_loss_.item()
                    total_reward_loss += reward_loss_.item()
                    total_loss += step_loss.item()

                    #print(f"[t={t}] stacked shape = {4 * t}, attention_mask = {input_attention_mask.shape[1]}")
                    self.global_step += 1

                #print("Batch Losses: Act: {:.4f}, State: {:.4f}, Return: {:.4f}".format())

                # Gradient step after the full sequence
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparams.max_grad_norm)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()

                avg_loss = total_loss / (T - 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.6f}")

                if (epoch * len(self.data_loader) + self.global_step) % self.hyperparams.log_interval == 0:
                    print(f"[Step {self.global_step}] Avg Loss: {avg_loss:.6f} | Act: {total_act_loss / (T - 1):.4f}, "
                        f"State: {total_state_loss / (T - 1):.4f}, Return: {total_reward_loss / (T - 1):.4f}")

                # Step the scheduler
                self.scheduler.step()

                epoch_loss += avg_loss
                progress_bar.set_postfix(loss=f"{avg_loss:.6f}")

            print(f"Epoch {epoch+1}/{self.hyperparams.epochs}, Loss: {epoch_loss / len(self.data_loader):.6f}")
            self.save_model(f"emt_model_epoch_{epoch+1}.pth")


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
