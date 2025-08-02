import os
import threading
import time
import numpy as np
import pygame
import torch

# Try to import stable_baselines3 models
try:
    from stable_baselines3 import DQN, PPO, A2C
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable_baselines3 not available. Some features may not work.")
    SB3_AVAILABLE = False
    DQN = PPO = A2C = None

# Import your custom modules with error handling
try:
    from environment.custom_env import GDMEnvironment, GDMRiskLevel, ActionType
    CUSTOM_ENV_AVAILABLE = True
except ImportError:
    print("Warning: Custom environment not fully available")
    CUSTOM_ENV_AVAILABLE = False
    # Define minimal fallbacks
    from enum import IntEnum
    
    class GDMRiskLevel(IntEnum):
        LOW = 0
        MODERATE = 1
        HIGH = 2
        CRITICAL = 3
    
    class ActionType(IntEnum):
        ROUTINE_MONITORING = 0
        INCREASED_MONITORING = 1
        GLUCOSE_TOLERANCE_TEST = 2
        CONTINUOUS_GLUCOSE_MONITORING = 3
        DIETARY_COUNSELING = 4
        EXERCISE_PROGRAM = 5
        WEIGHT_MANAGEMENT = 6
        STRESS_REDUCTION = 7
        INSULIN_THERAPY = 8
        METFORMIN_PRESCRIPTION = 9
        SPECIALIST_REFERRAL = 10
        IMMEDIATE_INTERVENTION = 11
        COMPREHENSIVE_ASSESSMENT = 12
        FAMILY_HISTORY_REVIEW = 13
        NO_ACTION = 14

from environment.rendering import ClearGDMVisualization

# Try to import training modules
try:
    from training.dqn_training import train_dqn
    DQN_TRAINING_AVAILABLE = True
except ImportError:
    print("Warning: DQN training not available")
    DQN_TRAINING_AVAILABLE = False

try:
    from training.pg_training import train_ppo, train_a2c, train_reinforce
    PG_TRAINING_AVAILABLE = True
except ImportError:
    print("Warning: Policy gradient training not available")
    PG_TRAINING_AVAILABLE = False

try:
    from evaluate import evaluate_model, evaluate_comparison
    EVALUATE_AVAILABLE = True
except ImportError:
    print("Warning: Evaluation modules not available")
    EVALUATE_AVAILABLE = False

# Define a simple REINFORCE policy class for loading
class ReinforcePolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, act_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

def demo_with_real_agent(viz, model=None, model_type="Random"):
    """Demo with agent (random or trained model) making decisions"""
    print(f"\nStarting visualization for {model_type} Agent")
    
    # Use the visualization's built-in demo method
    viz.demo_with_real_agent(model, model_type)

def run_interactive_visualization():
    """Run interactive visualization with manual control"""
    viz = ClearGDMVisualization()
    viz.run_visualization()

def load_model_safely(model_path, model_type, model_class, exp_name):
    """Safely load a model with error handling"""
    try:
        if model_type == "REINFORCE":
            if not os.path.exists(model_path):
                print(f"REINFORCE model not found at {model_path}")
                return None
            model = ReinforcePolicy(obs_dim=3, act_dim=15)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        else:
            if not SB3_AVAILABLE:
                print(f"stable_baselines3 not available for {model_type}")
                return None
            if not os.path.exists(model_path):
                print(f"{model_type} model not found at {model_path}")
                return None
            if not CUSTOM_ENV_AVAILABLE:
                print("Custom environment not available for model loading")
                return None
            
            env = GDMEnvironment()
            model = model_class.load(model_path, env=env)
            return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None

def main():
    print("🏥 GDM Care Journey - RL Summative")
    print("\nChoose mode:")
    print("1. Train DQN" + (" (Not Available)" if not DQN_TRAINING_AVAILABLE else ""))
    print("2. Train Policy Gradient Models (PPO, A2C, REINFORCE)" + (" (Not Available)" if not PG_TRAINING_AVAILABLE else ""))
    print("3. Visualize Trained Model")
    print("4. Run Interactive Visualization (Untrained, Manual)")
    print("5. Run Automated Untrained Simulation (Random Agent)")
    print("6. Evaluate Trained Model" + (" (Not Available)" if not EVALUATE_AVAILABLE else ""))
    
    choice = input("Enter choice (1, 2, 3, 4, 5, 6): ").strip()
    
    if choice in ["1", "2", "3", "6"]:  # These require experiment name
        exp_name = input("Enter experiment name (e.g., exp1): ").strip()
        if not exp_name:
            exp_name = "default"
        
        # Create directories
        os.makedirs("models/dqn", exist_ok=True)
        os.makedirs("models/pg", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("graph", exist_ok=True)
    
    if choice == "1":
        if not DQN_TRAINING_AVAILABLE:
            print("DQN training is not available. Please install required dependencies.")
            return
        print(f"Training DQN for experiment '{exp_name}'...")
        train_dqn(exp_name=exp_name)
        
    elif choice == "2":
        if not PG_TRAINING_AVAILABLE:
            print("Policy gradient training is not available. Please install required dependencies.")
            return
        print(f"Training Policy Gradient models for experiment '{exp_name}'...")
        
        try:
            ppo_result = train_ppo(exp_name=exp_name)
            print(f"PPO training completed: Mean Reward = {ppo_result[0]:.2f}, Std = {ppo_result[1]:.2f}")
        except Exception as e:
            print(f"PPO training failed: {e}")
        
        try:
            a2c_result = train_a2c(exp_name=exp_name)
            print(f"A2C training completed: Mean Reward = {a2c_result[0]:.2f}, Std = {a2c_result[1]:.2f}")
        except Exception as e:
            print(f"A2C training failed: {e}")
        
        try:
            reinforce_result = train_reinforce(exp_name=exp_name)
            print(f"REINFORCE training completed: Mean Reward = {reinforce_result[0]:.2f}, Std = {reinforce_result[1]:.2f}")
        except Exception as e:
            print(f"REINFORCE training failed: {e}")
            
    elif choice == "3":
        print("Select model to visualize:")
        print("1. DQN")
        print("2. PPO")
        print("3. A2C")
        print("4. REINFORCE")
        print("5. Random Agent (no model)")
        
        model_choice = input("Enter choice (1, 2, 3, 4, 5): ").strip()
        
        model_paths = {
            "1": (f"models/dqn/dqn_gdm_{exp_name}.zip", "DQN", DQN),
            "2": (f"models/pg/ppo_gdm_{exp_name}.zip", "PPO", PPO),
            "3": (f"models/pg/a2c_gdm_{exp_name}.zip", "A2C", A2C),
            "4": (f"models/pg/reinforce_gdm_{exp_name}.pth", "REINFORCE", None),
            "5": (None, "Random", None)
        }
        
        if model_choice in model_paths:
            model_path, model_type, model_class = model_paths[model_choice]
            
            if model_choice == "5":  # Random agent
                model = None
            else:
                model = load_model_safely(model_path, model_type, model_class, exp_name)
                if model is None:
                    print(f"Failed to load {model_type} model. Using random agent instead.")
                    model_type = "Random"
            
            try:
                viz = ClearGDMVisualization()
                demo_with_real_agent(viz, model, model_type)
            except Exception as e:
                print(f"Error in visualization: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid model choice.")
            
    elif choice == "4":
        print("Starting Interactive Visualization...")
        try:
            viz = ClearGDMVisualization()
            viz.run_visualization()
        except Exception as e:
            print(f"Error in interactive visualization: {e}")
            import traceback
            traceback.print_exc()
            
    elif choice == "5":
        print("Starting Automated Untrained Simulation...")
        try:
            viz = ClearGDMVisualization()
            demo_with_real_agent(viz, None, "Random")
        except Exception as e:
            print(f"Error in automated simulation: {e}")
            import traceback
            traceback.print_exc()
            
    elif choice == "6":
        if not EVALUATE_AVAILABLE:
            print("Evaluation is not available. Please check your evaluation modules.")
            return
            
        print("Select model to evaluate:")
        print("1. DQN")
        print("2. PPO") 
        print("3. A2C")
        print("4. REINFORCE")
        print("5. Compare All (DQN, PPO, A2C, REINFORCE)")
        
        model_choice = input("Enter choice (1, 2, 3, 4, 5): ").strip()
        
        algo_map = {
            "1": "dqn",
            "2": "ppo", 
            "3": "a2c",
            "4": "reinforce"
        }
        
        if model_choice in algo_map:
            algo_name = algo_map[model_choice]
            try:
                # Extract experiment number from exp_name
                exp_num = int(exp_name.replace("exp", "")) if exp_name.startswith("exp") else 1
                evaluate_model(exp_num, algo_name)
            except Exception as e:
                print(f"Error evaluating {algo_name.upper()} experiment {exp_name}: {e}")
                
        elif model_choice == "5":
            try:
                exp_num = int(exp_name.replace("exp", "")) if exp_name.startswith("exp") else 1
                evaluate_comparison(exp_num)
            except Exception as e:
                print(f"Error generating comparison for experiment {exp_name}: {e}")
        else:
            print("Invalid model choice.")
            
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()