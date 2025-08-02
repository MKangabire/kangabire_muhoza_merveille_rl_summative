import gym
import numpy as np
from enum import IntEnum
from gym.spaces import Discrete, Box
from environment.utils import get_risk_level, GDMRiskLevel, ActionType

class GDMEnvironment(gym.Env):
    def __init__(self, grid_size=8):
        super(GDMEnvironment, self).__init__()
        self.grid_size = grid_size
        self.action_space = Discrete(len(ActionType))
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 12.0], dtype=np.float32),
            high=np.array([float(grid_size-1), float(grid_size-1), 40.0], dtype=np.float32),
            dtype=np.float32
        )
        self.danger_cells = [(1, 3), (2, 5), (4, 2), (5, 6), (6, 1)]
        self.risky_cells = [(0, 3), (1, 2), (1, 4), (2, 4), (2, 6), (3, 2), (4, 1), (4, 3), (5, 5), (6, 0), (6, 2)]
        self.reset()

    def reset(self):
        self.current_position = (0, 0)
        self.current_risk_level = GDMRiskLevel.LOW
        self.gestational_week = 12
        self.step_count = 0
        return self._get_observation()

    def _get_observation(self):
        row, col = self.current_position
        return np.array([row, col, self.gestational_week], dtype=np.float32)

    def calculate_reward(self, risk_level, action):
        base_reward = 0
        if risk_level == GDMRiskLevel.LOW:
            if action == ActionType.ROUTINE_MONITORING:
                base_reward = 5
            elif action in [ActionType.DIETARY_COUNSELING, ActionType.EXERCISE_PROGRAM]:
                base_reward = 3
            elif action in [ActionType.INSULIN_THERAPY, ActionType.IMMEDIATE_INTERVENTION]:
                base_reward = -3
            else:
                base_reward = 1
        elif risk_level == GDMRiskLevel.MODERATE:
            if action in [ActionType.INCREASED_MONITORING, ActionType.GLUCOSE_TOLERANCE_TEST]:
                base_reward = 5
            elif action in [ActionType.DIETARY_COUNSELING, ActionType.EXERCISE_PROGRAM]:
                base_reward = 4
            elif action == ActionType.ROUTINE_MONITORING:
                base_reward = -1
            else:
                base_reward = 2
        elif risk_level == GDMRiskLevel.HIGH:
            if action in [ActionType.CONTINUOUS_GLUCOSE_MONITORING, ActionType.SPECIALIST_REFERRAL]:
                base_reward = 6
            elif action in [ActionType.INSULIN_THERAPY, ActionType.METFORMIN_PRESCRIPTION]:
                base_reward = 5
            elif action == ActionType.ROUTINE_MONITORING:
                base_reward = -4
            else:
                base_reward = 2
        elif risk_level == GDMRiskLevel.CRITICAL:
            if action == ActionType.IMMEDIATE_INTERVENTION:
                base_reward = 8
            elif action in [ActionType.INSULIN_THERAPY, ActionType.SPECIALIST_REFERRAL]:
                base_reward = 6
            elif action in [ActionType.ROUTINE_MONITORING, ActionType.NO_ACTION]:
                base_reward = -8
            else:
                base_reward = 3
        noise = np.random.normal(0, 1)
        return base_reward + noise

    def choose_next_position(self, current_pos, risk_level, action, step, gestational_week):
        row, col = current_pos
        possible_moves = []
        moves = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1), (row, col)]
        valid_moves = [(r, c) for r, c in moves if 0 <= r < self.grid_size and 0 <= c < self.grid_size]
        if risk_level == GDMRiskLevel.CRITICAL:
            if action in [ActionType.IMMEDIATE_INTERVENTION, ActionType.INSULIN_THERAPY]:
                if row > 0:
                    possible_moves.append((row - 1, col))
            else:
                possible_moves.extend([(row + 1, col + 1), (row, col + 1)] if row < 7 and col < 7 else [])
        elif action in [ActionType.ROUTINE_MONITORING, ActionType.INCREASED_MONITORING]:
            if risk_level == GDMRiskLevel.LOW:
                if col < 7:
                    possible_moves.append((row, col + 1))
                if row < 7 and col >= 4:
                    possible_moves.append((row + 1, col))
            else:
                possible_moves.append((row, col))
        elif action in [ActionType.DIETARY_COUNSELING, ActionType.EXERCISE_PROGRAM, ActionType.WEIGHT_MANAGEMENT]:
            if risk_level in [GDMRiskLevel.HIGH, GDMRiskLevel.MODERATE]:
                if row > 0:
                    possible_moves.append((row - 1, col + 1))
                if col < 7:
                    possible_moves.append((row, col + 1))
        elif action in [ActionType.INSULIN_THERAPY, ActionType.METFORMIN_PRESCRIPTION]:
            if row < 7:
                possible_moves.append((row + 1, col))
            if col < 7:
                possible_moves.append((row, col + 1))
        elif action == ActionType.SPECIALIST_REFERRAL:
            if row < 6 and col < 6:
                possible_moves.append((row + 2, col + 1))
            elif row > 1:
                possible_moves.append((row - 1, col + 1))
        if gestational_week > 35:
            if row < 7:
                possible_moves.append((row + 1, col))
            if col < 7:
                possible_moves.append((row, col + 1))
            if row < 7 and col < 7:
                possible_moves.append((row + 1, col + 1))
        if not possible_moves:
            possible_moves = valid_moves
        if np.random.random() < 0.3:
            possible_moves.extend(valid_moves)
        possible_moves = list(set(possible_moves))
        if gestational_week > 30:
            goal_moves = [(r, c) for r, c in possible_moves if r >= row and c >= col]
            if goal_moves:
                possible_moves = goal_moves
        return possible_moves[np.random.randint(len(possible_moves))] if possible_moves else current_pos

    def step(self, action):
        self.step_count += 1
        self.current_risk_level = get_risk_level(self.gestational_week, None)
        reward = self.calculate_reward(self.current_risk_level, action)
        self.current_position = self.choose_next_position(self.current_position, self.current_risk_level, action, self.step_count, self.gestational_week)
        self.gestational_week = min(40, self.gestational_week + np.random.randint(1, 3))
        done = self.current_position == (7, 7) or self.gestational_week >= 40 or self.step_count >= 20
        return self._get_observation(), reward, done, {"risk_level": self.current_risk_level.name}
