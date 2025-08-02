from stable_baselines3 import PPO, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import GDMEnvironment
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class CSVLoggingCallback(BaseCallback):
    """Callback to log mean reward and episode length to CSV for PPO and A2C"""
    def __init__(self, exp_name, algo_name, verbose=0):
        super(CSVLoggingCallback, self).__init__(verbose)
        try:
            self.experiment_number = int(exp_name.replace("exp", ""))
        except ValueError:
            self.experiment_number = 0
        self.algo_name = algo_name
        self.csv_log_path = f"logs/training_log_{algo_name.lower()}_{self.experiment_number}.csv"
        self.log_data = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self):
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            episode_reward = sum(self.locals["rewards"])
            episode_length = self.locals["n_steps"]
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if self.episode_count % 100 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_episode_length = np.mean(self.episode_lengths[-100:])
                self.log_data.append({
                    "timestep": self.num_timesteps,
                    "mean_reward_last_100": mean_reward,
                    "mean_episode_length_last_100": mean_episode_length
                })
                self.logger.record(f"{self.algo_name}/mean_reward_last_100", mean_reward)
                self.logger.record(f"{self.algo_name}/mean_episode_length_last_100", mean_episode_length)
                self.logger.dump(self.num_timesteps)
                print(f"{self.algo_name} Episode {self.episode_count}: Mean Reward = {mean_reward:.2f}, Mean Episode Length = {mean_episode_length:.2f}")
        
        return True

    def _on_training_end(self):
        if self.log_data:
            pd.DataFrame(self.log_data).to_csv(self.csv_log_path, index=False)
            print(f"CSV log saved: {self.csv_log_path}")

def train_ppo(exp_name="default"):
    """Train a PPO model with experiment-specific naming and CSV logging"""
    print(f"Starting PPO training for experiment '{exp_name}'...")
    log_path = f"logs/pg_{exp_name}.log"
    logger = configure(log_path, ["stdout", "log"])
    
    print("Initializing environment...")
    env = GDMEnvironment()
    print("Environment initialized.")
    
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    model.set_logger(logger)
    print("PPO model created.")
    
    total_timesteps = 100000
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, log_interval=100, callback=CSVLoggingCallback(exp_name, "PPO"))
    print(f"PPO training completed in {time.time() - start_time:.2f} seconds.")
    
    model_path = f"models/pg/ppo_gdm_{exp_name}"
    model.save(model_path)
    print(f"PPO model saved to {model_path}.zip")
    
    print("Evaluating PPO model...")
    rewards = []
    episodes = 10
    for i in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        logger.record(f"PPO/evaluation_episode_{i+1}", episode_reward)
    logger.dump(step=total_timesteps)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    logger.record("PPO/evaluation_mean_reward", mean_reward)
    logger.record("PPO/evaluation_std_reward", std_reward)
    logger.dump(step=total_timesteps)
    print(f"PPO Evaluation - Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    
    return mean_reward, std_reward

def train_a2c(exp_name="default"):
    """Train an A2C model with experiment-specific naming and CSV logging"""
    print(f"Starting A2C training for experiment '{exp_name}'...")
    log_path = f"logs/pg_{exp_name}.log"
    logger = configure(log_path, ["stdout", "log"])
    
    print("Initializing environment...")
    env = GDMEnvironment()
    print("Environment initialized.")
    
    print("Creating A2C model...")
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=20,
        gamma=0.95,
        gae_lambda=0.9,
        vf_coef=0.25,
        ent_coef=0.01,
        verbose=1
    )
    model.set_logger(logger)
    print("A2C model created.")
    
    total_timesteps = 100000
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, log_interval=100, callback=CSVLoggingCallback(exp_name, "A2C"))
    print(f"A2C training completed in {time.time() - start_time:.2f} seconds.")
    
    model_path = f"models/pg/a2c_gdm_{exp_name}"
    model.save(model_path)
    print(f"A2C model saved to {model_path}.zip")
    
    print("Evaluating A2C model...")
    rewards = []
    episodes = 10
    for i in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        logger.record(f"A2C/evaluation_episode_{i+1}", episode_reward)
    logger.dump(step=total_timesteps)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    logger.record("A2C/evaluation_mean_reward", mean_reward)
    logger.record("A2C/evaluation_std_reward", std_reward)
    logger.dump(step=total_timesteps)
    print(f"A2C Evaluation - Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    
    return mean_reward, std_reward

class ReinforcePolicy(nn.Module):
    """Simple MLP policy for REINFORCE"""
    def __init__(self, obs_dim, act_dim):
        super(ReinforcePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),  # Input: obs_dim=3 -> 128
            nn.ReLU(),
            nn.Linear(128, 64),       # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, act_dim),   # 64 -> act_dim=15
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension
        return self.net(obs)

def train_reinforce(exp_name="default"):
    """Train a custom REINFORCE model with experiment-specific naming and CSV logging"""
    try:
        experiment_number = int(exp_name.replace("exp", ""))
    except ValueError:
        experiment_number = 0
    print(f"Starting custom REINFORCE training for experiment '{exp_name}' (number {experiment_number})...")
    
    log_path = f"logs/pg_{exp_name}.log"
    csv_log_path = f"logs/training_log_reinforce_{experiment_number}.csv"
    logger = configure(log_path, ["stdout", "log"])
    
    print("Initializing environment...")
    env = GDMEnvironment()
    print("Environment initialized.")
    
    print("Creating REINFORCE policy...")
    obs_dim = env.observation_space.shape[0]  # Should be 3
    act_dim = env.action_space.n  # Should be 15
    print(f"Observation dimension: {obs_dim}, Action dimension: {act_dim}")
    torch.manual_seed(0)  # Ensure reproducible initialization
    policy = ReinforcePolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.0007)
    gamma = 0.95
    print("REINFORCE policy created.")
    
    total_timesteps = 200000
    episodes = 0
    timesteps = 0
    episode_rewards = []
    episode_lengths = []
    log_data = []
    print(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    
    while timesteps < total_timesteps:
        obs = env.reset()
        print(f"Episode {episodes + 1}: Initial observation shape: {obs.shape}")
        episode_log_probs = []
        episode_rewards_current = []
        episode_length = 0
        done = False
        
        while not done:
            obs_tensor = torch.FloatTensor(obs)
            print(f"Step {timesteps + 1}: obs_tensor shape: {obs_tensor.shape}")
            action_probs = policy(obs_tensor)
            print(f"Action probs shape: {action_probs.shape}")
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_obs, reward, done, _ = env.step(action.item())
            episode_log_probs.append(log_prob)
            episode_rewards_current.append(reward)
            episode_length += 1
            obs = next_obs
            timesteps += 1
            
            if timesteps >= total_timesteps:
                break
        
        episodes += 1
        episode_rewards.append(sum(episode_rewards_current))
        episode_lengths.append(episode_length)
        
        returns = []
        R = 0
        for r in episode_rewards_current[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_loss = []
        for log_prob, R in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if episodes % 100 == 0:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_episode_length = np.mean(episode_lengths[-100:])
            log_data.append({
                "timestep": timesteps,
                "mean_reward_last_100": mean_reward,
                "mean_episode_length_last_100": mean_episode_length
            })
            logger.record("REINFORCE/mean_reward_last_100", mean_reward)
            logger.record("REINFORCE/mean_episode_length_last_100", mean_episode_length)
            logger.dump(step=timesteps)
            print(f"REINFORCE Episode {episodes}: Mean Reward = {mean_reward:.2f}, Mean Episode Length = {mean_episode_length:.2f}")
    
    if log_data:
        pd.DataFrame(log_data).to_csv(csv_log_path, index=False)
        print(f"CSV log saved: {csv_log_path}")
    
    print(f"REINFORCE training completed in {time.time() - start_time:.2f} seconds.")
    
    model_path = f"models/pg/reinforce_gdm_{exp_name}.pth"
    torch.save(policy.state_dict(), model_path)
    print(f"REINFORCE model saved to {model_path}")
    
    print("Evaluating REINFORCE model...")
    rewards = []
    episodes_eval = 10
    policy.eval()  # Set to evaluation mode
    for i in range(episodes_eval):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs)
            action_probs = policy(obs_tensor)
            action = torch.argmax(action_probs).item()
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        logger.record(f"REINFORCE/evaluation_episode_{i+1}", episode_reward)
    logger.dump(step=timesteps)
    
    mean_reward = sum(rewards) / len(rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
    logger.record("REINFORCE/evaluation_mean_reward", mean_reward)
    logger.record("REINFORCE/evaluation_std_reward", std_reward)
    logger.dump(step=timesteps)
    print(f"REINFORCE Evaluation - Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    
    return mean_reward, std_reward