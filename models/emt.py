import numpy as np
import torch
import torch.nn as nn
import transformers

from models.abstractModel import AbstractModel
from models.transformer import GPT2Model


class EMT(AbstractModel):
    """
    This model uses a GPT-like transformer to model sequences of:
    (Return_1, state_1, action_1, Return_2, state_2, action_2, ...)
    plus config if needed.

    If `state_tanh` or `action_tanh` is True, we treat the corresponding
    outputs as continuous (with a Tanh layer).
    If they are False, we treat them as categorical (with raw logits).

    During training:
      - For continuous outputs (Tanh), you'd typically use MSE.
      - For categorical outputs (logits), you'd typically use CrossEntropy.

    Args:
        state_dim (int): size of the state vector
        act_dim (int): size of the action vector
        config_dim (int): size of the config vector
        hidden_size (int): hidden size for the Transformer
        max_length (int, optional): maximum sequence length for positional embeddings
        max_ep_len (int, optional): max episode length for embedding timesteps
        action_tanh (bool): if True => action is continuous (Tanh), else categorical (logits)
        state_tanh (bool): if True => state is continuous (Tanh), else categorical (logits)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        config_dim,
        hidden_size,
        max_length=None,
        max_ep_len=8192,
        action_tanh=True,
        state_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, config_dim=config_dim, max_length=max_length)
        self.hidden_size = hidden_size
        self.config_dim = config_dim

        config = transformers.GPT2Config(
            vocab_size=1,         # not actually used for tokens
            n_embd=hidden_size, 
            **kwargs
        )
        # A GPT2 model with custom positional embeddings
        self.transformer = GPT2Model(config)

        self.state_is_discrete = not state_tanh
        self.action_is_discrete = not action_tanh
        
        # Embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return   = nn.Linear(1, hidden_size)
        if self.state_is_discrete:
            print("Using embding ie deiscraete for states")
            self.embed_state = nn.Embedding(state_dim, hidden_size)
        else:
            print("Using Linear ie deiscraete for states")
            self.embed_state = nn.Linear(state_dim, hidden_size)

        if self.action_is_discrete:
            print("Using embding ie deiscraete for actions")
            self.embed_action = nn.Embedding(act_dim, hidden_size)
        else:
            print("Using Linear ie deiscraete for actions")
            self.embed_action = nn.Linear(act_dim, hidden_size)
        
        self.embed_config   = nn.Linear(self.config_dim, hidden_size)
        self.embed_ln       = nn.LayerNorm(hidden_size)

        # Output heads for states
        if state_tanh:
            # Continuous => No Tanh (direct regression)
            self.predict_state = nn.Linear(hidden_size, state_dim)
        else:
            # Categorical => raw logits
            self.predict_state = nn.Linear(hidden_size, state_dim)

        # Output heads for actions
        if action_tanh:
            # Continuous => No Tanh (direct regression)
            self.predict_action = nn.Linear(hidden_size, act_dim)
        else:
            # Categorical => raw logits
            self.predict_action = nn.Linear(hidden_size, act_dim)

        # Return regression head (usually continuous)
        self.predict_return = nn.Linear(hidden_size, 1)

        self.state_tanh = state_tanh
        self.action_tanh = action_tanh

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        config,
        attention_mask=None
    ):
        """
        Forward pass through the model.

        Shapes (assuming B = batch, T = seq_length):
            states:        [B, T, state_dim]
            actions:       [B, T, act_dim]
            returns_to_go: [B, T, 1]
            timesteps:     [B, T] (for embedding)
            config:        [B, T, config_dim]
            attention_mask:[B, T], optional

        Returns:
            state_preds:  [B, T, state_dim]  (categorical logits or continuous Tanh)
            action_preds: [B, T, act_dim]    (categorical logits or continuous Tanh)
            return_preds: [B, T, 1]          (continuous, typically MSE)
        """

        batch_size, seq_length = states.shape[0], states.shape[1]

        # Default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), 
                dtype=torch.long, 
                device=states.device
            )

        # print("State Size:", states.size(), states.dtype) 
        # print("Action Size:", actions.size(), actions.dtype)
        # print("Config Size", config.size(), config.dtype)
        # 1) Embed each modality
        if self.state_is_discrete:
            states = states.argmax(dim=-1) 
            state_embeddings = self.embed_state(states)
        else:
            # states are real vectors of shape [B, T, state_dim]
            state_embeddings = self.embed_state(states.float())

        if self.action_is_discrete:
            actions = actions.argmax(dim=-1)
            action_embeddings = self.embed_action(actions)
        else:
            # states are real vectors of shape [B, T, state_dim]
            action_embeddings = self.embed_action(actions.float())

        #state_embeddings   = self.embed_state(states.float())             # [B, T, H]
        #action_embeddings  = self.embed_action(actions.float())           # [B, T, H]
        config_embeddings  = self.embed_config(config.float())            # [B, T, H]
        returns_embeddings = self.embed_return(returns_to_go.float())     # [B, T, H]
        time_embeddings    = self.embed_timestep(timesteps)               # [B, T, H]

        # print("State Emb Size:", state_embeddings.size(), state_embeddings.dtype) 
        # print("Action Emb Size:", action_embeddings.size(), action_embeddings.dtype)
        # print("Config Emb Size", config_embeddings.size(), config_embeddings.dtype)
       

        # 2) Add positional (time) embeddings
        state_embeddings   += time_embeddings
        action_embeddings  += time_embeddings
        config_embeddings  += time_embeddings
        returns_embeddings += time_embeddings

        # 3) Stack in order (returns, state, action, config) => shape [B, T, 4, H], flatten => [B, 4T, H]
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings, config_embeddings), dim=2)
            .reshape(batch_size, 4 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # 4) Construct an attention mask in the same shape => [B, 4T]
        stacked_attention_mask = attention_mask.unsqueeze(1)              # [B, 1, T]
        stacked_attention_mask = stacked_attention_mask.expand(-1, 4, -1) # [B, 4, T]
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, 4 * seq_length)  # [B, 4*T]
        
        # print("Sizes of the stuff", stacked_inputs.size(), stacked_attention_mask.size())

        # 5) Pass through the GPT2 transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=False
        )
        x = transformer_outputs['last_hidden_state']  # [B, 4T, H]

        # 6) Reshape back => [B, T, 4, H]
        x = x.reshape(batch_size, seq_length, 4, self.hidden_size)
        # reorder to x[:,0] -> returns, x[:,1] -> state, x[:,2] -> action, x[:,3] -> config
        x = x.permute(0, 2, 1, 3)  # => [B, 4, T, H]

        # 7) Extract predicted values:
        #    - next returns from the 'action' token  (x[:,2])
        #    - next state   from the 'action' token  (x[:,2]) or 'state' token? (Arbitrary choice, but consistent)
        #    - next action  from the 'state' token   (x[:,1])
        #    Adjust to your preference if needed.
        return_preds = self.predict_return(x[:, 2])   # [B, T, 1]
        state_preds  = self.predict_state(x[:, 2])    # [B, T, state_dim]
        action_preds = self.predict_action(x[:, 1])   # [B, T, act_dim]

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        configs,
        **kwargs
    ):
        """
        Used for inference: returns the predicted action at the last timestep.

        If `action_tanh` = True => output is continuous (Tanh).
        If `action_tanh` = False => output is raw logits for CrossEntropy.
        """

        # Reshape inputs to [1, T, ...] for a single batch
        states       = states.reshape(1, -1, self.state_dim)
        actions      = actions.reshape(1, -1, self.act_dim)
        returns_to_go= returns_to_go.reshape(1, -1, 1)
        timesteps    = timesteps.reshape(1, -1)
        configs      = configs.reshape(1, -1, self.config_dim)

        seq_length = states.shape[1]

        if self.max_length is not None:
            # If we have a max_length, we apply truncation and then pad
            states       = states[:, -self.max_length:]
            actions      = actions[:, -self.max_length:]
            returns_to_go= returns_to_go[:, -self.max_length:]
            timesteps    = timesteps[:, -self.max_length:]
            configs      = configs[:, -self.max_length:]
            seq_length   = states.shape[1]

            # Construct the attention mask
            attention_mask = torch.cat([
                torch.zeros(self.max_length - seq_length, device=states.device),
                torch.ones(seq_length, device=states.device)
            ], dim=0).long().reshape(1, -1)

            # Helper to pad to max_length
            def pad_to(seq, total, dim_size):
                pad_size = total - seq.size(1)
                if pad_size <= 0:
                    return seq
                pad_shape = list(seq.shape)
                pad_shape[1] = pad_size
                return torch.cat([torch.zeros(*pad_shape, device=seq.device), seq], dim=1)

            states       = pad_to(states,       self.max_length, self.state_dim)
            actions      = pad_to(actions,      self.max_length, self.act_dim)
            returns_to_go= pad_to(returns_to_go,self.max_length, 1)
            timesteps    = pad_to(timesteps,    self.max_length, 1)
            configs      = pad_to(configs,      self.max_length, self.config_dim)
        else:
            attention_mask = None

        # Forward pass
        state_preds, action_preds, return_preds = self.forward(
            states,
            actions,
            rewards=None,  # not used
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            config=configs,
            attention_mask=attention_mask,
            **kwargs
        )

        # Return the last predicted action
        # shape of action_preds is [B=1, T, act_dim]
        # If `self.action_tanh` == False => raw logits
        # If `self.action_tanh` == True  => continuous (Tanh)
        return action_preds[0, -1]
