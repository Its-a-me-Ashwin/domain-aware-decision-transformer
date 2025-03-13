import numpy as np
import gym
from gym import spaces
import pygame

class GridWorldConfig:
    def __init__(self, action_inverter=False, slip_factor=0.0, wormhole_state=0, max_steps=1000):
        self.action_inverter = action_inverter  # If True, mirror actions
        self.slip_factor = slip_factor  # Probability of slipping to a random adjacent cell
        self.wormhole_state = wormhole_state  # -1 (send to start), 1 (send near goal), 0 (random teleport)
        self.max_steps = max_steps

    @staticmethod
    def generate_random_config():
        """Generates a random environment configuration."""
        return GridWorldConfig(
            action_inverter=np.random.choice([True, False]),  # Randomly enable/disable action inversion
            slip_factor=np.random.uniform(0, 0.3),  # Random slip probability
            wormhole_state=np.random.uniform(-1, 1),  # Random wormhole placement control
            max_steps=1000 + np.random.randint(0, 1000)
        )
    
    def to_dict(self):
        """Returns a dictionary representation of the configuration with JSON-safe types."""
        return {
            "action_inverter": bool(self.action_inverter),  # Convert np.bool_ → Python bool
            "slip_factor": float(self.slip_factor),  # Convert np.float32/64 → Python float
            "wormhole_state": float(self.wormhole_state),  # Ensure float type
            "max_steps": int(self.max_steps)  # Ensure integer type
        }
    
    def config_to_vector(self):
        param_order = [
            "action_inverter",
            "slip_factor",
            "wormhole_state",
            "max_steps"
        ]
        return [1 if self.action_inverter else 0, self.slip_factor, self.wormhole_state, self.max_steps ]



class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=GridWorldConfig(), gui=False):
        super(GridWorldEnv, self).__init__()
        
        self.grid_size = 10
        self.start_pos = (0, 0)
        self.target_1 = (4, 4)
        self.target_2 = (9, 9)
        self.agent_pos = list(self.start_pos)
        self.gui = gui
        self.config = config
        self.max_reward = 0
        self.steps = 0
        
        # Define wormhole positions
        self.wormhole_start_pos = self._set_wormhole_positions(-self.config.wormhole_state)
        self.wormhole_end_pos = self._set_wormhole_positions(self.config.wormhole_state)

        # Action Space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        self.action_dim = 1
        self.sate_dim = 2
        self.config_dim = 4
        
        # Observation Space: Agent's position in the grid
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        # Pygame Init
        if self.gui:
            pygame.init()
            self.screen = pygame.display.set_mode((500, 500))
            self.clock = pygame.time.Clock()

    def _set_wormhole_positions(self, state):
        """ Determines where the wormhole is placed based on configuration. """
        skew_factor = 10  # Controls how skewed the distribution is

        def sample_skewed_position(target):
            """ Samples a position skewed towards the given target. """
            x = min(int(np.random.gamma(shape=skew_factor, scale=1.5)), 9)
            y = min(int(np.random.gamma(shape=skew_factor, scale=1.5)), 9)
            if target == (9,9):
                return (9 - x, 9 - y)  # Flip distribution to be near (9,9)
            else:
                return (x, y)  # Bias towards (0,0)

        def adjust_position(pos):
            """ Ensures that (0,0), (4,4), and (9,9) are avoided. """
            forbidden = {(0,0), (4,4), (9,9)}
            if pos in forbidden:
                return (pos[0] + 1 if pos[0] < 9 else pos[0] - 1, 
                        pos[1] + 1 if pos[1] < 9 else pos[1] - 1)
            return pos

        if state == 0:
            # Fully random wormhole location avoiding (0,0), (4,4), (9,9)
            while True:
                pos = (np.random.randint(10), np.random.randint(10))
                if pos not in {(0,0), (4,4), (9,9)}:
                    return pos
        
        elif state <= -1:
            return adjust_position((9, 9))  # Closest to (9,9) but adjusted
        
        elif state >= 1:
            return adjust_position((0, 0))  # Closest to (0,0) but adjusted
        
        else:
            # Interpolate position based on state
            target = (9,9) if state < 0 else (0,0)
            pos = sample_skewed_position(target)
            return adjust_position(pos)

    
    def step(self, action):
        if self.config.action_inverter:
            try:
                ## For numpy stuff.
                action = {0: 1, 1: 0, 2: 3, 3: 2}[action.astype(int).item()]
            except:
                action = {0: 1, 1: 0, 2: 3, 3: 2}[action]
        self.steps += 1
        intended_pos = list(self.agent_pos)
        if action == 0: intended_pos[1] -= 1  # Up
        elif action == 1: intended_pos[1] += 1  # Down
        elif action == 2: intended_pos[0] -= 1  # Left
        elif action == 3: intended_pos[0] += 1  # Right
        
        # Slip effect
        if np.random.rand() < self.config.slip_factor:
            intended_pos = [
                max(0, min(self.grid_size - 1, self.agent_pos[0] + np.random.choice([-1, 0, 1]))),
                max(0, min(self.grid_size - 1, self.agent_pos[1] + np.random.choice([-1, 0, 1])))
            ]
        

        # Ensure within bounds
        prev_pos = self.agent_pos.copy()

        # Clamp the agent position within bounds
        self.agent_pos = [
            max(0, min(self.grid_size - 1, intended_pos[0])),
            max(0, min(self.grid_size - 1, intended_pos[1]))
        ]

        # Check if the intended move was out of bounds
        out_of_bounds = (prev_pos[0] != intended_pos[0] and intended_pos[0] != self.agent_pos[0]) or \
                        (prev_pos[1] != intended_pos[1] and intended_pos[1] != self.agent_pos[1])


        # Rewards
        reward = -100 if out_of_bounds else -1 
        done = False
        if tuple(self.agent_pos) == self.target_1:
            reward = 10
            self.max_reward += 1
        elif tuple(self.agent_pos) == self.target_2:
            reward = 100
            self.max_reward += 1

        if self.max_reward == 2 or self.steps > self.config.max_steps or out_of_bounds:    
            done = True
        
        # Wormhole Effect
        if tuple(self.agent_pos) == self.wormhole_start_pos:
            self.agent_pos = list(self.wormhole_end_pos)
            
        ### obs, reward, done, truncated, info
        return np.array(self.agent_pos), reward, done, False, {}

    
    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agent_pos = list(self.start_pos)
        self.max_reward = False
        self.wormhole_start_pos = self._set_wormhole_positions(-self.config.wormhole_state)
        self.wormhole_end_pos = self._set_wormhole_positions(self.config.wormhole_state)
        self.steps = 0
        return np.array(self.agent_pos), {}
    
    def render(self, mode='human'):
        if not self.gui:
            return
        
        self.screen.fill((255, 255, 255))
        cell_size = 50
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        
        # Draw targets
        pygame.draw.rect(self.screen, (0, 255, 0), (self.target_1[0] * cell_size, self.target_1[1] * cell_size, cell_size, cell_size))
        pygame.draw.rect(self.screen, (0, 0, 255), (self.target_2[0] * cell_size, self.target_2[1] * cell_size, cell_size, cell_size))
        
        # Draw wormhole
        if self.wormhole_start_pos:
            pygame.draw.rect(self.screen, (255, 255, 0), (self.wormhole_start_pos[0] * cell_size, self.wormhole_start_pos[1] * cell_size, cell_size, cell_size))
        if self.wormhole_end_pos:
            pygame.draw.rect(self.screen, (0, 0, 0), (self.wormhole_end_pos[0] * cell_size, self.wormhole_end_pos[1] * cell_size, cell_size, cell_size))
        
        # Draw agent
        pygame.draw.rect(self.screen, (255, 0, 0), (self.agent_pos[0] * cell_size, self.agent_pos[1] * cell_size, cell_size, cell_size))
        
        pygame.display.flip()
        self.clock.tick(10)
    
    def close(self):
        if self.gui:
            pygame.quit()

    def play_keyboard(self):
        """ Allows playing the environment using keyboard controls. """
        if not self.gui:
            print("GUI is disabled. Enable it to use keyboard controls.")
            return
        
        running = True
        done = False
        reward = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        obs, reward, done, _, _ = self.step(0)
                    elif event.key == pygame.K_DOWN:
                        obs, reward, done, _, _ = self.step(1)
                    elif event.key == pygame.K_LEFT:
                        obs, reward, done, _, _ = self.step(2)
                    elif event.key == pygame.K_RIGHT:
                        obs, reward, done, _, _ = self.step(3)
            #print(reward, done)
            self.render()
        self.close()
    
# Example Usage
if __name__ == "__main__":
    config = GridWorldConfig(action_inverter=False, slip_factor=0.0, wormhole_state=-0.9)
    env = GridWorldEnv(config=config, gui=True)
    env.play_keyboard()
