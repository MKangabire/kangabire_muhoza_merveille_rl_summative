# GDM Care Journey RL System

## Project Overview
This project implements a reinforcement learning (RL) system to optimize care pathways for patients with Gestational Diabetes Mellitus (GDM). The system simulates an intelligent decision-making agent that navigates a patient through a grid-based care journey, selecting actions to manage health risks and maximize positive health outcomes while minimizing complications across gestational weeks (12–40).

## Problem Context
Based on the challenges in managing GDM, patients face risks due to:
- **Variable Health Risks**: Fluctuating risk levels (low, moderate, high, critical) based on gestational weeks and health metrics.
- **Complex Decision-Making**: Balancing monitoring, medical interventions, and lifestyle changes to prevent complications like high-risk pregnancies or preterm delivery.
- **Resource Constraints**: Limited access to timely interventions in resource-constrained settings, requiring efficient action prioritization.
- **Dynamic Progression**: Need for adaptive care plans as patient conditions evolve over time, similar to navigating a dynamic environment like `FrozenLake`.

## Solution Approach
The RL agent operates within a custom Gymnasium environment (`GDMEnvironment`) where:
- **States**: Patient health state represented by a 3D vector (`[row, col, gestational_week]`), capturing position in a care journey grid and pregnancy progression (12–40 weeks).
- **Actions**: 15 discrete actions (`Discrete(15)`), including monitoring (e.g., glucose checks), medical interventions (e.g., insulin therapy), lifestyle interventions (e.g., diet changes), and other care actions.
- **Rewards**: Based on reducing health risks, achieving healthy delivery (goal state), and avoiding negative outcomes (e.g., critical risk escalation). Rewards are calculated using risk levels (`GDMRiskLevel.LOW`, `MODERATE`, `HIGH`, `CRITICAL`) and action effectiveness.

The system supports four RL algorithms:
- **DQN**: Learns optimal action-value functions for discrete actions.
- **PPO**: Optimizes policies with clipped objectives for stability.
- **A2C**: Combines actor-critic learning for efficient training.
- **REINFORCE**: Custom policy gradient with a wider MLP architecture (`nn.Linear(3, 128)`, `nn.Linear(128, 64)`, `nn.Linear(64, 15)`) to capture complex care patterns.

Training metrics are logged to CSV files (`logs/training_log_<algo>_<exp_number>.csv`), and a 3D visualization tool using Pygame and OpenGL displays patient movement, risk levels, actions, and rewards.
## Project structure

kangabire_muhoza_merveille_rl_summative/
├── environment/
│   ├── custom_env.py       # GDMEnvironment implementation
│   ├── rendering.py        # 3D visualization using Pygame/OpenGL
│   └── utils.py            # Enums (GDMRiskLevel, ActionType) and reward calculation
├── training/
│   ├── dqn_training.py     # DQN training script
│   └── pg_training.py      # PPO, A2C, REINFORCE training with CSV logging
├── logs/                   # Training logs (e.g., training_log_reinforce_3.csv)
├── models/                 # Saved models (e.g., reinforce_gdm_exp3.pth, ppo_gdm_exp1.zip)
├── graph/                  # Generated reward and episode length graphs
├── evaluate_reinforce.py   # Script to visualize training metrics
└── main.py                 # Main script for training, visualization, and simulation


## Requirements
- Python 3.10.11
- Dependencies (install via `requirements.txt` or manually):
  ```bash
  pip install pandas torch stable-baselines3 numpy gym matplotlib pygame

### Installation
## 1. Clone the Repository
 git clone <repository-url>
 cd kangabire_muhoza_merveille_rl_summative
 Set Up Virtual Environment:
bashpython -m venv rl
source rl/bin/activate  # On Windows: rl\Scripts\activate

Install Dependencies:
bashpip install pandas torch stable-baselines3 numpy gym matplotlib pygame
Or, if provided:
bashpip install -r requirements.txt

Verify Pygame:

Ensure Pygame 2.6.1 or compatible is installed for visualization.
Note: A warning about pkg_resources deprecation may appear but does not affect functionality. To suppress, update setuptools:
bashpip install --upgrade setuptools




### Usage
    Run the main script to train, visualize, or simulate:
    bashpython main.py
    Menu Options
    text🏥 GDM Care Journey - RL Summative
    Choose mode:
    1. Train DQN
    2. Train Policy Gradient Models (PPO, A2C, REINFORCE)
    3. Visualize Trained Model
    4. Run Interactive Visualization (Untrained, Manual)
    5. Run Automated Untrained Simulation (Random Agent)
    Enter choice (1, 2, 3, 4, 5):

## Option 1: Train DQN

    Trains a DQN model and saves to models/pg/dqn_gdm_expX.zip.
    Logs to logs/training_log_dqn_X.csv.
    Example:
    textEnter experiment name (e.g., exp1): exp1



## Option 2: Train Policy Gradient Models

    Trains PPO, A2C, and REINFORCE for a given experiment.
    Saves models to models/pg/<algo>_gdm_expX.zip (PPO, A2C) or .pth (REINFORCE).
    Logs to logs/training_log_<algo>_X.csv.
    REINFORCE uses a custom MLP: nn.Linear(3, 128), nn.Linear(128, 64), nn.Linear(64, 15).
    Example:
    textEnter experiment name (e.g., exp1): exp3



## Option 3: Visualize Trained Model

    Visualizes a trained model’s behavior in 3D.
    Supports DQN, PPO, A2C, REINFORCE.
    Example:
    textEnter experiment name (e.g., exp1): exp1
    Select model to visualize:
    1. DQN
    2. PPO
    3. A2C
    4. REINFORCE
    Enter choice (1, 2, 3, 4): 1



## Option 4: Interactive Visualization (Untrained, Manual)

    Runs an interactive simulation where you control actions via keyboard.
    No model required.


## Option 5: Automated Untrained Simulation (Random Agent)

    Runs a simulation with a random agent.



### Visualization Controls

Mouse Drag: Rotate view
Mouse Wheel: Zoom in/out
SPACE: Simulate next step
R: Reset camera view
ESC: Exit

## Visual Elements

Grid Cells: Light Blue (safe), Yellow (risky), Red (danger), Green (goal).
Patient (Sphere): Green (low risk), Yellow (moderate), Orange (high), Red (critical).
Actions (Shapes): Blue cube (monitoring), Red sphere (medical), Green cube (lifestyle), Yellow sphere (other).
Rewards: Green cubes (positive), Red cubes (negative).
Progress Bar: Green bar showing gestational weeks (12–40).

## Outputs

Training Logs: CSV files in logs/ (e.g., training_log_reinforce_3.csv) with columns:

timestep
mean_reward_last_100
mean_episode_length_last_100


Models: Saved in models/pg/ (e.g., reinforce_gdm_exp3.pth, dqn_gdm_exp1.zip).
Graphs: Reward and episode length plots in graph/ (e.g., reinforce_reward_experiment_3.png).
### 🧪 DQN Hyperparameter Configurations

| Hyperparameter              | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 |
|----------------------------|--------------|--------------|--------------|--------------|
| **Learning Rate**          | 0.0003       | 0.0001       | 0.0002       | 0.0004       |
| **Gamma**                  | 0.99         | 0.95         | 0.95         | 0.95         |
| **Batch Size**             | 64           | 64           | 64           | 128          |
| **Buffer Size**            | 50,000       | 50,000       | 50,000       | 50,000       |
| **Initial Epsilon**        | 1.0          | 1.0          | 1.0          | 1.0          |
| **Final Epsilon**          | 0.05         | 0.02         | 0.02         | 0.01         |
| **Exploration Fraction**   | 0.3          | 0.3          | 0.3          | 0.3          |
| **Target Update Interval** | 10,000       | 10,000       | 10,000       | 10,000       |
| **Train Frequency**        | 4            | 4            | 4            | 4            |
| **Learning Starts**        | 1,000        | 1,000        | 1,000        | 1,000        |
| **Gradient Steps**         | 1            | 1            | 1            | 1            |
| **Total Timesteps**        | 100,000      | 100,000      | 300,000      | 300,000      |

*Observation*
## DQN 1
Used a learning rate of 0.0003, gamma 0.99, batch size 64, and replay buffer of 10,000. The agent started with an exploration fraction of 0.2, decaying epsilon to 0.02.
Mean reward showed a gradual and steady improvement in the early stages of training, increasing from around 2.47 at timestep 1,839 to a local peak of 3.95 around 20,000. Afterward, the performance fluctuated, with the reward oscillating between ~3.0 and 3.7, and occasionally dropping below 3.0.
Notably, mean reward peaked again at timestep 95,443 (≈3.92), indicating partial recovery towards the end, but the reward curve never fully stabilized, suggesting the agent had not yet converged.

Episode length remained relatively stable between 16.5 and 18.5, showing consistent episode structure but no strong correlation with reward spikes.
The larger batch size (64) may have contributed to smoother learning but slower updates, while the low learning rate (0.0003) helped avoid instability but may have slowed convergence. A longer training period or modified exploration schedule (e.g., slower decay) could help the agent achieve better final performance.
## DQN 2
The first experiment, with a learning rate of 0.0003 and gamma of 0.99, showed steady improvement in mean reward, peaking around timestep 20,000 and fluctuating throughout training, with partial recovery toward the end. In contrast, the second experiment used a lower learning rate (0.0001), a reduced gamma (0.95), and a larger replay buffer (50,000). While it achieved a slightly earlier reward peak, the agent’s performance dropped sharply mid-training and remained unstable, recovering intermittently. The lower gamma and slower learning likely contributed to less stable long-term behavior, while the large buffer may have introduced learning delays. Overall, both models showed promise but did not fully converge within 100,000 timesteps, with the first experiment demonstrating slightly more consistent learning progression.
## DQN 3
Across the three experiments, the third model delivered the most successful training outcome. By adjusting the learning rate to 0.0002 and extending training to 300,000 timesteps, it achieved sustained high rewards (above 4.0) with low fluctuation. In contrast, the first experiment (with learning rate 0.0003 and gamma 0.99) showed moderate early progress but lacked long-term consistency, while the second experiment (lower learning rate of 0.0001 and gamma 0.95) fluctuated throughout and struggled to stabilize. The third model benefited from balanced hyperparameters and extended exposure, leading to more stable learning and better convergence—clearly outperforming the others in both reward maximization and policy consistency.

## DQN 4
Compared to the previous three experiments, this fourth trial achieved the best overall performance. With a slightly more aggressive learning rate of 0.0004 and a larger batch size of 128, the model converged faster and more consistently. Rewards surpassed previous peaks, reaching 4.48 and maintaining that level for a significant portion of the training. Unlike Experiment 1 and 2, which exhibited instability or early plateaus, this model benefited from improved hyperparameters and extended training time. It even outperformed Experiment 3—which already showed strong learning—by delivering smoother convergence and better final rewards. The reduced exploration rate (epsilon) also likely helped fine-tune policy exploitation in the later stages.

## 🟥 REINFORCE

 
| **Hyperparameter**    | **Experiment 1**     | **Experiment 2**      | **Experiment 3**     |
| --------------------- | -------------------- | --------------------- | -------------------- |
| Learning Rate         | 0.0003               | 0.0007                | 0.0010               |
| Gamma                 | 0.99                 | 0.95                  | 0.98                 |
| Optimizer             | Adam                 | Adam                  | RMSprop              |
| Policy Architecture   | MLP (64-64-act\_dim) | MLP (128-64-act\_dim) | MLP (64-64-act\_dim) |
| Total Timesteps       | 100,000              | 200,000               | 100,000              |
| Activation Function   | ReLU                 | ReLU                  | ReLU                 |
| Output Activation     | Softmax              | Softmax               | Softmax              |
| Observation Dimension | 3                    | 3                     | 3                    |
| Action Dimension      | 15                   | 15                    | 15                   |
| Random Seed           | 0                    | 0                     | Not specified        |

## Reinforce 1
Utilized a learning rate of 0.0003, gamma 0.99, and a custom MLP policy with layers of 64 units. The agent exhibited significant reward improvement early, starting at 50.48 and peaking at 68.39 around 89,862 timesteps, but fluctuated between 62 and 68 thereafter, ending at 64.10. Episode lengths remained stable around 17 steps, with minor variations (16.35 to 18.18). The reward curve showed initial convergence but later instability, suggesting the learning rate or policy architecture may need adjustment. Extending training or implementing learning rate scheduling could enhance stability and push rewards higher.
## REINFORCE 2
Employed a learning rate of 0.0007, gamma 0.95, and a modified MLP policy with layers of 128 and 64 units, trained for 200,000 timesteps. Mean rewards started at 51.95, peaked at 56.61 around 20,332 timesteps, but dropped sharply to ~43–47 after 25,643 timesteps, ending at 44.36. Episode lengths remained stable at ~16–18.5 steps. Compared to the previous setup (rewards 50–68, ending at 64.10), the higher learning rate and larger MLP likely caused instability and overfitting, leading to a significant reward decline. Adjusting the learning rate downward or reverting to the original MLP architecture could restore stability and improve performance.
## Reinforce 3
Utilized a learning rate of 0.0010, gamma 0.98, RMSprop optimizer, and an MLP policy with 64-64-act_dim layers, trained for 100,000 timesteps. Mean rewards started at 54.78, peaked at 68.87 at 67,613 timesteps, and ended at 64.25, showing strong early gains but later fluctuations. Episode lengths remained stable at ~16.5–18.3 steps. Compared to Experiment 1 (rewards 50.48–68.39, end 64.10) and Experiment 2 (42.94–56.61, end 44.36), the RMSprop optimizer and reverted MLP restored performance to Experiment 1’s level, but the high learning rate caused fluctuations, suggesting a need for scheduling or a lower rate to achieve full convergence.

## 🟩 A2C


| **Hyperparameter**  | **Experiment 1** | **Experiment 2** | **Experiment 3** |
| ------------------- | ---------------- | ---------------- | ---------------- |
| Learning Rate       | 0.0007           | 0.0003           | 0.001            |
| Gamma               | 0.99             | 0.95             | 0.98             |
| n\_steps            | 5                | 20               | 10               |
| gae\_lambda         | 1.0              | 0.9              | 0.95             |
| vf\_coef            | 0.5              | 0.25             | 0.4              |
| ent\_coef           | 0.00             | 0.01             | 0.005            |
| Policy Architecture | MlpPolicy        | MlpPolicy        | MlpPolicy        |
| Total Timesteps     | 100,000          | 100,000          | 100,000          |
| Verbose             | 1                | 1                | 1                |

## A2C 1
Employed a learning rate of 0.0007, gamma 0.99, n_steps of 5, and an MlpPolicy. The agent struggled significantly, with mean rewards fluctuating narrowly between 2.61 and 4.19, ending at 3.08 after 100,000 timesteps, showing no clear improvement. Episode lengths were consistently short, averaging 1.9–2.2 steps, indicating frequent early terminations. The lack of reward progress and short episodes suggest the learning rate may be too high or n_steps too low for effective learning. Increasing n_steps or revising the policy architecture could improve performance.
## A2C 2
Utilized a learning rate of 0.0003, gamma 0.95, n_steps of 20, gae_lambda of 0.9, vf_coef of 0.25, and ent_coef of 0.01 with an MlpPolicy. Mean rewards started at 2.83, peaked at 4.35 at 15,616 timesteps, and ended at 4.17, showing slight improvement but no strong trend. Episode lengths increased to ~7.69–11.06 steps, averaging ~9–10, compared to ~1.9–2.2 previously. The lower learning rate and higher n_steps improved exploration over the prior setup (rewards 2.61–4.19, ending at 3.08), but low rewards suggest the policy remains suboptimal, requiring further tuning of n_steps or a custom policy.
## A2C 3
Employed a learning rate of 0.001, gamma 0.98, n_steps of 10, gae_lambda of 0.95, vf_coef of 0.4, and ent_coef of 0.005 with an MlpPolicy. Mean rewards started at 2.53, peaked at 3.56 at 29,699 timesteps, and ended at 2.89, with no clear upward trend. Episode lengths averaged ~4–5 steps, ranging from 3.93 to 5.33. Compared to Experiment 1 (rewards 2.61–4.19, end 3.08, lengths ~1.66–2.33) and Experiment 2 (2.83–4.35, end 4.17, lengths ~7.69–11.06), the higher learning rate and reduced n_steps led to worse performance, with shorter episodes indicating early terminations and a need for increased n_steps or a lower learning rate.

## 🟦 PPO

| **Hyperparameter**  | **Experiment 1** | **Experiment 2** | **Experiment 3** |
| ------------------- | ---------------- | ---------------- | ---------------- |
| Learning Rate       | 0.0003           | 0.0001           | 0.0003           |
| Gamma               | 0.99             | 0.99             | 0.95             |
| n\_steps            | 2048             | 2048             | 1024             |
| Batch Size          | 64               | 64               | 32               |
| n\_epochs           | 10               | 10               | 5                |
| gae\_lambda         | 0.95             | 0.95             | 0.90             |
| clip\_range         | 0.2              | 0.2              | 0.3              |
| Policy Architecture | MlpPolicy        | MlpPolicy        | MlpPolicy        |
| Total Timesteps     | 100,000          | 100,000          | 100,000          |
| Verbose             | 1                | 1                | 1                |

## PPO1
Used a learning rate of 0.0003, gamma 0.99, n_steps of 2048, batch size of 64, and clip_range of 0.2 with an MlpPolicy. Mean rewards started at 1.65, gradually increased to a peak of 3.62 at 84,292 timesteps, and ended at 3.58, showing modest progress. Episode lengths were unusually long, ranging from 903.96 to 1143.34 steps, with no clear trend. The reward curve indicated slow learning without full convergence, possibly due to conservative exploration or suboptimal reward scaling. Adjusting clip_range or reward normalization could enhance reward accumulation.
## PPO2
Used a learning rate of 0.0001, gamma 0.99, n_steps of 2048, batch size of 64, and clip_range of 0.2 with an MlpPolicy. Mean rewards started at 1.73, peaked at 4.52 at 90,742 timesteps, and ended at 4.51, showing steady improvement. Episode lengths remained long, ranging from 857.31 to 1194.64 steps, similar to the previous setup (~900–1143). Compared to prior results (rewards 1.65–3.62, ending at 3.58), the lower learning rate enhanced stability and reward growth, but long episodes and moderate fluctuations suggest conservative exploration. Adjusting clip_range or reward scaling could further optimize performance.
## PPO 3
Employed a learning rate of 0.001, gamma 0.98, n_steps of 10, gae_lambda of 0.95, vf_coef of 0.4, and ent_coef of 0.005 with an MlpPolicy. Mean rewards started at 2.53, peaked at 3.56 at 29,699 timesteps, and ended at 2.89, with no clear upward trend. Episode lengths averaged ~4–5 steps, ranging from 3.93 to 5.33. Compared to Experiment 1 (rewards 2.61–4.19, end 3.08, lengths ~1.66–2.33) and Experiment 2 (2.83–4.35, end 4.17, lengths ~7.69–11.06), the higher learning rate and reduced n_steps led to worse performance, with shorter episodes indicating early terminations and a need for increased n_steps or a lower learning rate.
