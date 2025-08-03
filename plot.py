import pandas as pd
import matplotlib.pyplot as plt
import os

# Define log file paths based on experiment number
def get_log_path(exp_num, model):
    exp_number = int(exp_num.replace("exp", ""))
    if model.lower() == "reinforce":
        return f"C:\\Users\\Merveille\\kangabire_muhoza_merveille_rl_summative\\logs\\training_log_{model.lower()}_{exp_number}.csv"
    return f"C:\\Users\\Merveille\\kangabire_muhoza_merveille_rl_summative\\logs\\training_log_{model.lower()}_{exp_number}.csv"

# Create plots directory if it doesn't exist
plots_dir = r"C:\Users\Merveille\kangabire_muhoza_merveille_rl_summative\plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# Input from user
exp_num = input("Enter experiment number (e.g., exp1): ")
model = input("Enter model (DQN, PPO, A2C, or REINFORCE): ").upper()

# Load the log file
log_path = get_log_path(exp_num, model)
if not os.path.exists(log_path):
    print(f"Error: Log file {log_path} not found. Please ensure the model was trained for {exp_num}.")
    exit()

df = pd.read_csv(log_path)

# Plot 1: Cumulative Reward
plt.figure(figsize=(10, 6))
plt.plot(df['timestep'], df['mean_reward_last_100'], label=f'{model} Cumulative Reward', color='blue')
plt.xlabel('Timestep')
plt.ylabel('Mean Reward (Last 100 Episodes)')
plt.title(f'Cumulative Reward for {model} - Experiment {exp_num}')
plt.legend()
plt.grid(True)
plt.savefig(f"{plots_dir}\\cumulative_reward_{model}_{exp_num}.png")
plt.show()

# Plot 2: Training Stability (Using Reward as Proxy)
plt.figure(figsize=(10, 6))
plt.plot(df['timestep'], df['mean_reward_last_100'], label=f'{model} Reward Stability', color='green')
plt.xlabel('Timestep')
plt.ylabel('Mean Reward (Last 100 Episodes)')
plt.title(f'Training Stability for {model} - Experiment {exp_num}')
plt.legend()
plt.grid(True)
plt.savefig(f"{plots_dir}\\training_stability_{model}_{exp_num}.png")
plt.show()

print(f"Plots saved as 'cumulative_reward_{model}_{exp_num}.png' and 'training_stability_{model}_{exp_num}.png' in the plots folder.")