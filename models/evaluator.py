import torch

class Evaluator:
    def __init__(self, env, model, K, device):
        """
        Args:
            env: The environment to interact with.
            model: The Decision Transformer model (EMT).
            K: Context length for autoregressive evaluation.
            device: Torch device ('cpu' or 'cuda').
        """
        self.env = env
        self.model = model
        self.K = K
        self.device = device

    def evaluate(self, target_return=25, is_action_discreate=False, is_state_discreate=False):
        """
        Runs an evaluation episode using autoregressive sampling.

        Args:
            target_return (float): The initial return-to-go for expert-level performance.
        
        Returns:
            list: Collected states, actions, rewards, and returns.
        """
        # Initialize episode
        state, _ = self.env.reset()

        R = [torch.tensor(target_return, dtype=torch.float32, device=self.device)]  # Returns-to-go
        s = [torch.tensor(state, dtype=torch.float32, device=self.device)]  # Initial state
        a = []  # no Actions taken
        t = [torch.tensor(1, dtype=torch.long, device=self.device)]  # Timesteps
        done = False

        # Get the config vector from the environment
        config = torch.tensor(self.env.config.config_to_vector(), dtype=torch.float32, device=self.device)

        ## Batch size is 1 [B, CTX, dim] B is 1 all the time. 
        while not done:
            # Prepare model input tensors (ensuring correct shapes)
            
            # Process states
            states = torch.stack([torch.tensor(state, dtype=torch.float32, device=self.device) for state in s]).unsqueeze(0)

            # Process actions
            if len(a) > 0:
                actions = torch.tensor([0] + a, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)  # [1, n, 1]
            else:
                actions = torch.zeros((1, 1, 1), dtype=torch.float32, device=self.device)  # [1, 1, 1]

            # Process returns-to-go (ensure [1, K, 1])
            returns_to_go = torch.stack(R[-self.K:]).unsqueeze(0).unsqueeze(-1).to(self.device)  # [1, K, 1]

            # Process timesteps
            timesteps = torch.stack(t[-self.K:]).unsqueeze(0).to(self.device)  # [1, K]

            # Process config (ensure correct shape)
            if len(s) > 1:  # If there are multiple states, repeat config
                configs = config.unsqueeze(0).expand(1, len(s), -1).to(self.device)  # [1, n, config_dim]
            else:
                configs = config.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, config_dim]

            # print("Invaluation loop:", 
            #     "states:", states.size(), 
            #     "actions:", actions.size(), 
            #     "returns_to_go:", returns_to_go.size(), 
            #     "configs:", configs.size())
            
            ## Forward 
            state_preds, action_preds, return_preds = self.model.forward(
                states, actions, None, returns_to_go, timesteps, configs
            )

            # Sample last predicted action
            #action = action_preds[:, -1, :].squeeze(0)  # Extract last action in sequence
            print(action_preds)
            action = action_preds[0, -1]
            print(action)
            # Step in environment
            new_s, r, done, _, _ = self.env.step(action.cpu().detach().numpy())
            # Append new tokens to sequence
            R.append(R[-1] - r)  # Update returns-to-go
            s.append(torch.tensor(new_s, dtype=torch.float32, device=self.device))
            a.append(action)
            t.append(torch.tensor(len(R), dtype=torch.long, device=self.device))

            # Keep only the last K tokens
            R, s, a, t = R[-self.K:], s[-self.K:], a[-self.K:], t[-self.K:]

        return s, a, R
