from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import GDMEnvironment
import os
import numpy as np
import pandas as pd

class CSVLoggingCallback(BaseCallback):
    """Callback to log mean reward and episode length to CSV for DQN"""
    def __init__(self, exp_name, verbose=0):
        super(CSVLoggingCallback, self).__init__(verbose)
        try:
            self.experiment_number = int(exp_name.replace("exp", ""))
        except ValueError:
            self.experiment_number = 0
        self.csv_log_path = f"logs/training_log_dqn_{self.experiment_number}.csv"
        self.log_data = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_length = 0  # Track steps in current episode

    def _on_step(self):
        self.current_episode_length += 1  # Increment step count
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            episode_reward = self.locals["rewards"][0]  # Single environment, so index 0
            episode_length = self.current_episode_length
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.current_episode_length = 0  # Reset for next episode
            
            if self.episode_count % 100 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_episode_length = np.mean(self.episode_lengths[-100:])
                self.log_data.append({
                    "timestep": self.num_timesteps,
                    "mean_reward_last_100": mean_reward,
                    "mean_episode_length_last_100": mean_episode_length
                })
                self.logger.record("DQN/mean_reward_last_100", mean_reward)
                self.logger.record("DQN/mean_episode_length_last_100", mean_episode_length)
                self.logger.dump(self.num_timesteps)
                print(f"DQN Episode {self.episode_count}: Mean Reward = {mean_reward:.2f}, Mean Episode Length = {mean_episode_length:.2f}")
        
        return True

    def _on_training_end(self):
        if self.log_data:
            pd.DataFrame(self.log_data).to_csv(self.csv_log_path, index=False)
            print(f"CSV log saved: {self.csv_log_path}")

def train_dqn(exp_name="default"):
    """Train a DQN model for the GDM environment with experiment-specific naming and CSV logging"""
    print(f"Starting DQN training for experiment '{exp_name}'...")
    log_path = f"logs/dqn_{exp_name}.log"
    logger = configure(log_path, ["stdout", "log"])
    
    print("Creating environment...")
    env = DummyVecEnv([lambda: GDMEnvironment()])  # Wrap in DummyVecEnv
    print("Environment created.")
    
    print("Initializing DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0004,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        tau=1.0,
        gamma=0.95,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        verbose=1
    )
    model.set_logger(logger)
    print("DQN model initialized.")
    
    total_timesteps = 300000
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, log_interval=100, callback=CSVLoggingCallback(exp_name))
    print("DQN training completed.")
    
    model_path = f"models/dqn/dqn_gdm_{exp_name}"
    model.save(model_path)
    print(f"DQN model saved to {model_path}.zip")
    
    print("Evaluating DQN model...")
    rewards = []
    episodes = 10
    for i in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0]  # Single environment, so index 0
        rewards.append(episode_reward)
        logger.record(f"DQN/evaluation_episode_{i+1}", episode_reward)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    logger.record("DQN/evaluation_mean_reward", mean_reward)
    logger.record("DQN/evaluation_std_reward", std_reward)
    logger.dump(step=total_timesteps)
    print(f"DQN Evaluation - Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    
    return mean_reward, std_reward