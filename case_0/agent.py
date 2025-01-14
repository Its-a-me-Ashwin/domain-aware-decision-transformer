from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.env_checker import check_env
import gym
import argparse
import time

# Suppose you have your custom environment:
from env import DoubleInvertedPendulumBalanceEnv

def train():
    env = DoubleInvertedPendulumBalanceEnv(use_gui=False, max_episode_steps=10000, max_force=50.0)
    check_env(env, warn=True)

    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     # You can tune these parameters:
    #     n_steps=2048,
    #     batch_size=64,
    #     ent_coef=0.0,
    #     learning_rate=3e-4,
    #     n_epochs=10,#     n_epochs=10,
    # )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,     # Good default for Adam optimizer
        buffer_size=10000,  # Replay buffer size
        batch_size=256,         # Batch size for each gradient update
        tau=0.005,              # Soft update coefficient (for target network)
        gamma=0.999,             # Discount factor
        train_freq=(1, "step"), # Update every 1 step
        gradient_steps=1,       # # of gradient updates after each train freq
        ent_coef="auto_0.1",    # Automatic entropy tuning (starting target entropy ~0.1)
        verbose=1,              # Print training info
        device="auto"           # Use GPU if available, else CPU
    )

    model.learn(total_timesteps=300_000)
    model.save("sac_double_inverted_pendulum")
    env.close()


def test():
    env = DoubleInvertedPendulumBalanceEnv(use_gui=True, max_episode_steps=10000)
    loaded_model = SAC.load("sac_double_inverted_pendulum", env=env)

    obs, _ = env.reset()
    for idx in range(10000):
        action, _ = loaded_model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        if done:
            obs, _ = env.reset()
        time.sleep(1./240.)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="method")
    parser.add_argument(
        "--train_test",
        type=int,
        choices=[0, 1],  # Only allow 0 or 1
        default=1,
        help="0 to train 1 to test",
    )

    args = parser.parse_args()
    if (args.train_test == 0):
        train()
    elif (args.train_test == 1):
        test()