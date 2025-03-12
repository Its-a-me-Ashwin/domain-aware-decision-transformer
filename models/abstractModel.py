import numpy as np
import torch
import torch.nn as nn

## Used as interface used to compare other models with the EMT
class AbstractModel(nn.Module):
    def __init__(self, state_dim, act_dim, config_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.config_Dim = config_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, config, masks=None, attention_mask=None):
        return None, None, None

    def get_action(self, states, actions, rewards, config, **kwargs):
        return torch.zeros_like(actions[-1])
    

