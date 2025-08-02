import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_model(experiment_number, algo_name):
    """
    Evaluate a model's training log and generate graphs.
    
    Args:
        experiment_number (int): The experiment number
        algo_name (str): Name of the algorithm (dqn, ppo, a2c, reinforce)
    """
    log_path = f"logs/training_log_{algo_name.lower()}_{experiment_number}.csv"
    
    if not os.path.exists(log_path):
        print(f"ERROR: Log file {log_path} not found!")
        print(f"Make sure you have trained the {algo_name.upper()} model for experiment {experiment_number}.")
        return None
    
    print(f"Reading training log: {log_path}")
    log = pd.read_csv(log_path)
    
    os.makedirs("graph", exist_ok=True)
    
    # Graph 1: Training Reward Trend
    plt.figure(figsize=(12, 6))
    plt.plot(log['timestep'], log['mean_reward_last_100'], 
             label=f'{algo_name.upper()} Experiment {experiment_number} - Mean Reward (Last 100 Episodes)',
             linewidth=2, 
             color='purple' if algo_name == 'dqn' else 'blue' if algo_name == 'ppo' else 'green' if algo_name == 'a2c' else 'red')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Mean Reward', fontsize=12)
    plt.title(f'{algo_name.upper()} Training Reward Trend - Experiment {experiment_number}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    reward_graph_path = f"graph/{algo_name.lower()}_reward_experiment_{experiment_number}.png"
    plt.savefig(reward_graph_path, dpi=300, bbox_inches='tight')
    print(f"Reward graph saved: {reward_graph_path}")
    plt.show()
    
    # Graph 2: Training Episode Length Trend
    plt.figure(figsize=(12, 6))
    plt.plot(log['timestep'], log['mean_episode_length_last_100'], 
             label=f'{algo_name.upper()} Experiment {experiment_number} - Mean Episode Length (Last 100 Episodes)',
             linewidth=2, 
             color='purple' if algo_name == 'dqn' else 'blue' if algo_name == 'ppo' else 'green' if algo_name == 'a2c' else 'red')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Mean Episode Length', fontsize=12)
    plt.title(f'{algo_name.upper()} Training Episode Length Trend - Experiment {experiment_number}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    episode_graph_path = f"graph/{algo_name.lower()}_episode_length_experiment_{experiment_number}.png"
    plt.savefig(episode_graph_path, dpi=300, bbox_inches='tight')
    print(f"Episode length graph saved: {episode_graph_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\n=== {algo_name.upper()} Experiment {experiment_number} Summary ===")
    print(f"Total timesteps: {log['timestep'].max():,}")
    print(f"Final mean reward: {log['mean_reward_last_100'].iloc[-1]:.2f}")
    print(f"Final mean episode length: {log['mean_episode_length_last_100'].iloc[-1]:.2f}")
    print(f"Best mean reward: {log['mean_reward_last_100'].max():.2f}")
    print(f"Best mean episode length: {log['mean_episode_length_last_100'].max():.2f}")
    
    return log

def evaluate_comparison(experiment_number):
    """
    Generate a comparison graph for mean rewards of DQN, PPO, A2C, and REINFORCE.
    
    Args:
        experiment_number (int): The experiment number
    """
    logs = {}
    colors = {'dqn': 'purple', 'ppo': 'blue', 'a2c': 'green', 'reinforce': 'red'}
    
    plt.figure(figsize=(12, 6))
    all_logs_exist = True
    for algo_name in ['dqn', 'ppo', 'a2c', 'reinforce']:
        log_path = f"logs/training_log_{algo_name.lower()}_{experiment_number}.csv"
        if not os.path.exists(log_path):
            print(f"ERROR: Log file {log_path} not found!")
            all_logs_exist = False
            continue
        log = pd.read_csv(log_path)
        logs[algo_name] = log
        plt.plot(log['timestep'], log['mean_reward_last_100'], 
                 label=f'{algo_name.upper()} - Mean Reward (Last 100 Episodes)',
                 linewidth=2, color=colors[algo_name])
    
    if all_logs_exist and logs:
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title(f'Comparison of Training Reward Trends - Experiment {experiment_number}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        comparison_graph_path = f"graph/comparison_reward_experiment_{experiment_number}.png"
        plt.savefig(comparison_graph_path, dpi=300, bbox_inches='tight')
        print(f"Comparison reward graph saved: {comparison_graph_path}")
        plt.show()
    else:
        print("Cannot generate comparison graph due to missing log files.")

def evaluate_all_experiments():
    """Evaluate all experiments for DQN, PPO, A2C, and REINFORCE."""
    print("=== Model Evaluation ===")
    
    for experiment in range(1, 5):  # Assuming up to 4 experiments
        print(f"\n--- Evaluating Experiment {experiment} ---")
        for algo_name in ['dqn', 'ppo', 'a2c', 'reinforce']:
            print(f"\nEvaluating {algo_name.upper()}...")
            try:
                evaluate_model(experiment, algo_name)
            except Exception as e:
                print(f"Error evaluating {algo_name.upper()} experiment {experiment}: {e}")
                print("Skipping to next algorithm...")
        print(f"\nGenerating comparison graph for Experiment {experiment}...")
        try:
            evaluate_comparison(experiment)
        except Exception as e:
            print(f"Error generating comparison graph for experiment {experiment}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        experiment_num = int(sys.argv[1])
        algo_name = sys.argv[2].lower() if len(sys.argv) > 2 else None
        if 1 <= experiment_num <= 5 and algo_name in ['dqn', 'ppo', 'a2c', 'reinforce', None]:
            if algo_name:
                evaluate_model(experiment_num, algo_name)
            else:
                for algo in ['dqn', 'ppo', 'a2c', 'reinforce']:
                    evaluate_model(experiment_num, algo)
                evaluate_comparison(experiment_num)
        else:
            print("Experiment number must be between 1 and 5, and algorithm must be dqn, ppo, a2c, or reinforce")
    else:
        evaluate_all_experiments()
