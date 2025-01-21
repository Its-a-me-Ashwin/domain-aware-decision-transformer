import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, size=(1,))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor).detach().numpy()
        return act_values[0]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = self.model(state_tensor).detach().clone()
            if done:
                target[0][0] = reward
            else:
                t = self.target_model(next_state_tensor).detach()
                try:
                    #print(target[0][0], reward + self.gamma * torch.max(t).item())
                    target[0][0] = reward + self.gamma * torch.max(t).item()
                    #print(target[0][0])
                except TypeError:
                    #print(target[0][0], reward + self.gamma * torch.max(t).item())
                    temp = (reward + self.gamma * torch.max(t).item())[0]
                    target[0][0] = torch.tensor(temp, dtype=torch.float32)
                except Exception as e:
                    print("Error in parsing the reward:", e)
                    break
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    from env import InvertedPendulumBalanceEnv

    env = InvertedPendulumBalanceEnv(use_gui=False)
    state_size = env.observation_space.shape[0]
    action_size = 1
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 64

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time_t in tqdm(range(500)):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time_t}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.update_target_model()

    torch.save(agent.model.state_dict(), "dqn_inverted_pendulum_model.pth")
    print("Model saved as dqn_inverted_pendulum_model.pth")

    env.close()
