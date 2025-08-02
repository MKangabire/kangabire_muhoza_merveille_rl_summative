import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
from typing import Dict, List, Tuple
import threading
from enum import IntEnum

# Import your custom modules with fallbacks - EXACTLY as in your original code
try:
    from environment.utils import GDMRiskLevel, ActionType, get_risk_level
    from environment.custom_env import GDMEnvironment
except ImportError:
    # Fallback definitions - EXACTLY as in your original code
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

import torch
import pandas as pd

class CleanGDMVisualization:
    """Clean, beautiful 3D visualization - using your exact environment structure"""
    
    def __init__(self, grid_size=(8, 8), screen_size=(1200, 800)):
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.screen = None
        self.episode = 0
        self.total_reward = 0
        self.running = True
        self.position = (0, 0)
        self.risk_level = GDMRiskLevel.LOW
        self.action = ActionType.ROUTINE_MONITORING
        self.reward = 0
        self.gestational_week = 12
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Enhanced mouse control variables
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        self.rotation_x = 30
        self.rotation_y = 45
        self.zoom = 15
        
        # Visual elements
        self.patient_path = []
        self.rewards_history = []
        
        # EXACT action-to-visual mapping from your original code
        self.action_to_visual = {
            ActionType.ROUTINE_MONITORING: ('cube', (0.0, 0.0, 1.0, 0.7)),
            ActionType.INCREASED_MONITORING: ('cube', (0.0, 0.5, 0.5, 0.7)),
            ActionType.GLUCOSE_TOLERANCE_TEST: ('cube', (0.5, 0.0, 0.5, 0.7)),
            ActionType.CONTINUOUS_GLUCOSE_MONITORING: ('cube', (0.0, 0.5, 0.5, 0.7)),
            ActionType.DIETARY_COUNSELING: ('cube', (0.0, 1.0, 0.0, 0.7)),
            ActionType.EXERCISE_PROGRAM: ('sphere', (0.7, 0.0, 0.7, 0.7)),
            ActionType.WEIGHT_MANAGEMENT: ('cube', (0.5, 0.5, 0.0, 0.7)),
            ActionType.STRESS_REDUCTION: ('sphere', (0.7, 0.0, 0.7, 0.7)),
            ActionType.INSULIN_THERAPY: ('sphere', (1.0, 0.0, 0.0, 0.7)),
            ActionType.METFORMIN_PRESCRIPTION: ('cube', (1.0, 0.5, 0.0, 0.7)),
            ActionType.SPECIALIST_REFERRAL: ('cube', (0.5, 0.0, 0.5, 0.7)),
            ActionType.IMMEDIATE_INTERVENTION: ('cube', (0.8, 0.0, 0.0, 0.7)),
            ActionType.COMPREHENSIVE_ASSESSMENT: ('cube', (0.0, 0.8, 0.0, 0.7)),
            ActionType.FAMILY_HISTORY_REVIEW: ('cube', (0.0, 0.0, 0.8, 0.7)),
            ActionType.NO_ACTION: ('sphere', (0.3, 0.3, 0.3, 0.7)),
        }
        
        # Beautiful enhanced colors for better visual appeal
        self.enhanced_colors = {
            # Environment - more beautiful versions
            'background': (0.05, 0.05, 0.15),
            'grid_safe': (0.7, 0.9, 1.0, 0.8),      # Light blue ice
            'grid_risky': (1.0, 0.8, 0.4, 0.8),     # Amber warning
            'grid_danger': (1.0, 0.3, 0.3, 0.8),    # Red danger
            'grid_goal': (0.3, 1.0, 0.3, 0.8),      # Green goal
            'grid_lines': (0.2, 0.2, 0.4),
            
            # Patient colors - enhanced
            'patient_low': (0.2, 0.9, 0.2, 0.9),         # Healthy green
            'patient_moderate': (0.9, 0.7, 0.2, 0.9),     # Warning orange  
            'patient_high': (0.9, 0.4, 0.2, 0.9),        # Alert red-orange
            'patient_critical': (0.9, 0.1, 0.1, 0.9),    # Critical red
        }
        
        # Initialize pygame and OpenGL - enhanced but compatible
        pygame.init()
        pygame.display.set_caption("GDM Care Journey - Enhanced 3D Visualization")
        self.screen = pygame.display.set_mode(self.screen_size, pygame.OPENGL | pygame.DOUBLEBUF)
        
        # Initialize OpenGL with enhancements
        self.init_opengl()
    
    def init_opengl(self):
        """Initialize OpenGL with enhancements but same base as original"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Beautiful background
        bg = self.enhanced_colors['background']
        glClearColor(bg[0], bg[1], bg[2], 1.0)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.screen_size[0] / self.screen_size[1], 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def glutSolidSphere(self, radius, slices=20, stacks=20):
        """Replace GLUT sphere with custom implementation - EXACT as your original"""
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                glVertex3f(radius * x * zr0, radius * y * zr0, radius * z0)
                glVertex3f(radius * x * zr1, radius * y * zr1, radius * z1)
            glEnd()
    
    def glutSolidCube(self, size):
        """Replace GLUT cube with custom implementation - EXACT as your original"""
        s = size / 2
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-s, -s, s)
        glVertex3f(s, -s, s)
        glVertex3f(s, s, s)
        glVertex3f(-s, s, s)
        
        # Back face
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, -s, -s)
        
        # Top face
        glVertex3f(-s, s, -s)
        glVertex3f(-s, s, s)
        glVertex3f(s, s, s)
        glVertex3f(s, s, -s)
        
        # Bottom face
        glVertex3f(-s, -s, -s)
        glVertex3f(s, -s, -s)
        glVertex3f(s, -s, s)
        glVertex3f(-s, -s, s)
        
        # Right face
        glVertex3f(s, -s, -s)
        glVertex3f(s, s, -s)
        glVertex3f(s, s, s)
        glVertex3f(s, -s, s)
        
        # Left face
        glVertex3f(-s, -s, -s)
        glVertex3f(-s, -s, s)
        glVertex3f(-s, s, s)
        glVertex3f(-s, s, -s)
        
        glEnd()
    
    def print_instructions(self):
        """Print instructions - EXACTLY as in your original code"""
        print("\n============================================================\n"
              "GDM CARE JOURNEY - ENHANCED 3D VISUALIZATION\n"
              "(Clear and Simple like FrozenLake)\n"
              "============================================================\n\n"
              "CONTROLS:\n"
              "  Mouse Drag: Rotate view\n"
              "  Mouse Wheel: Zoom in/out\n"
              "  SPACE: Simulate next step\n"
              "  R: Reset camera view\n"
              "  ESC: Exit\n\n"
              "VISUAL ELEMENTS:\n"
              "  GRID CELLS (like FrozenLake ice):\n"
              "    • Light Blue: Safe (low risk)\n"
              "    • Yellow: Risky (moderate risk)\n"
              "    • Red: Danger (high risk)\n"
              "    • Green: Goal (healthy delivery)\n"
              "  PATIENT (colored sphere):\n"
              "    • Green: Low risk\n"
              "    • Yellow: Moderate risk\n"
              "    • Orange: High risk\n"
              "    • Red (pulsing): Critical risk\n"
              "  ACTIONS (shapes above patient):\n"
              "    • Blue Cube: Routine Monitoring\n"
              "    • Teal Cube: Increased Monitoring\n"
              "    • Magenta Cube: Glucose Tolerance Test\n"
              "    • Teal Cube: Continuous Glucose Monitoring\n"
              "    • Green Cube: Dietary Counseling\n"
              "    • Purple Sphere: Exercise Program\n"
              "    • Olive Cube: Weight Management\n"
              "    • Purple Sphere: Stress Reduction\n"
              "    • Red Sphere: Insulin Therapy\n"
              "    • Orange Cube: Metformin Prescription\n"
              "    • Magenta Cube: Specialist Referral\n"
              "    • Dark Red Cube: Immediate Intervention\n"
              "    • Dark Green Cube: Comprehensive Assessment\n"
              "    • Dark Blue Cube: Family History Review\n"
              "    • Gray Sphere: No Action\n"
              "  REWARDS (floating cubes):\n"
              "    • Green: Positive reward\n"
              "    • Red: Negative reward\n"
              "  PROGRESS BAR (green bar at bottom):\n"
              "    • Shows gestational week progress (12-40 weeks)\n\n"
              "The patient moves through the care journey like\n"
              "an agent moving through FrozenLake!\n"
              "============================================================\n")
    
    def reset_episode_stats(self, episode):
        """EXACTLY as your original"""
        with self.data_lock:
            self.episode = episode
            self.total_reward = 0
            self.patient_path = []
            self.rewards_history = []
    
    def update_state(self, position, risk_level, action, reward, gestational_week):
        """Update state - EXACTLY as your original but with thread safety"""
        with self.data_lock:
            # Handle numpy arrays properly - EXACT as your original
            if hasattr(position, '__len__') and len(position) >= 2:
                if hasattr(position[0], 'item'):  # numpy scalar
                    old_position = self.position
                    self.position = (float(position[0].item()), float(position[1].item()))
                else:
                    old_position = self.position
                    self.position = (float(position[0]), float(position[1]))
            else:
                old_position = self.position
                self.position = position
                
            self.risk_level = risk_level
            self.action = action
            self.reward = reward
            self.gestational_week = gestational_week
            self.total_reward += reward
            
            # Update path tracking
            if self.position != old_position and self.position not in self.patient_path:
                self.patient_path.append(self.position)
                
            # Keep path manageable
            if len(self.patient_path) > 25:
                self.patient_path = self.patient_path[-25:]
    
    def handle_events(self):
        """Handle events - enhanced but compatible with your original"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reset_camera()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                pass  # For manual step simulation
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_pressed = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Mouse wheel up
                    self.zoom = max(5, self.zoom - 1)
                elif event.button == 5:  # Mouse wheel down
                    self.zoom = min(30, self.zoom + 1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_pressed = False
            elif event.type == pygame.MOUSEMOTION and self.mouse_pressed:
                mouse_pos = pygame.mouse.get_pos()
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                self.rotation_y += dx * 0.5
                self.rotation_x += dy * 0.5
                self.rotation_x = max(-90, min(90, self.rotation_x))
                self.last_mouse_pos = mouse_pos
    
    def reset_camera(self):
        """EXACTLY as your original"""
        self.rotation_x = 30
        self.rotation_y = 45
        self.zoom = 15

    def draw_enhanced_grid(self):
        """Draw beautiful grid while maintaining original logic"""
        with self.data_lock:
            risk_level = self.risk_level
            
        # Enhanced risk colors
        risk_colors = {
            GDMRiskLevel.LOW: self.enhanced_colors['grid_safe'],
            GDMRiskLevel.MODERATE: self.enhanced_colors['grid_risky'], 
            GDMRiskLevel.HIGH: self.enhanced_colors['grid_danger'],
            GDMRiskLevel.CRITICAL: (0.5, 0.0, 0.0, 0.8)
        }
        
        # Draw grid cells with enhanced visuals
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # Goal cell special color
                if (x, y) == (7, 7):
                    color = self.enhanced_colors['grid_goal']
                else:
                    color = risk_colors.get(risk_level, self.enhanced_colors['grid_safe'])
                
                glColor4f(*color)
                
                # Draw enhanced cell as slightly raised quad
                glBegin(GL_QUADS)
                glVertex3f(x, y, 0.05)
                glVertex3f(x + 1, y, 0.05)
                glVertex3f(x + 1, y + 1, 0.05)
                glVertex3f(x, y + 1, 0.05)
                glEnd()
        
        # Draw beautiful grid lines
        glColor3f(*self.enhanced_colors['grid_lines'])
        glLineWidth(1.5)
        glBegin(GL_LINES)
        for x in range(self.grid_size[0] + 1):
            glVertex3f(x, 0, 0.1)
            glVertex3f(x, self.grid_size[1], 0.1)
        for y in range(self.grid_size[1] + 1):
            glVertex3f(0, y, 0.1)
            glVertex3f(self.grid_size[0], y, 0.1)
        glEnd()
        glLineWidth(1.0)

    def draw_patient_path(self):
        """Draw patient path as beautiful line"""
        with self.data_lock:
            path = self.patient_path.copy()
        
        if len(path) < 2:
            return
            
        glColor4f(0.8, 0.3, 0.8, 0.7)  # Purple with transparency
        glLineWidth(3.0)
        glBegin(GL_LINE_STRIP)
        
        for pos in path:
            x, y = pos
            glVertex3f(x + 0.5, y + 0.5, 0.3)
            
        glEnd()
        glLineWidth(1.0)
    
    def render(self):
        """Enhanced render function maintaining your original structure"""
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Apply camera transformations - enhanced but same logic
            glTranslatef(0, 0, -self.zoom)
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 0, 1)
            glTranslatef(-4, -4, 0)
            
            # Draw enhanced grid
            self.draw_enhanced_grid()
            
            # Draw patient path
            self.draw_patient_path()
            
            # Render patient sphere - enhanced but EXACTLY same logic as original
            with self.data_lock:
                risk_level = self.risk_level
                position = self.position
                
            patient_colors = {
                GDMRiskLevel.LOW: self.enhanced_colors['patient_low'],
                GDMRiskLevel.MODERATE: self.enhanced_colors['patient_moderate'],
                GDMRiskLevel.HIGH: self.enhanced_colors['patient_high'],
                GDMRiskLevel.CRITICAL: self.enhanced_colors['patient_critical']
            }
            
            color = patient_colors.get(risk_level, self.enhanced_colors['patient_low'])
            glColor4f(*color)
            glPushMatrix()
            glTranslatef(position[0] + 0.5, position[1] + 0.5, 0.5)
            
            # Pulsing effect for critical patients
            radius = 0.3
            if risk_level == GDMRiskLevel.CRITICAL:
                radius = 0.3 + 0.1 * math.sin(time.time() * 4)
                
            self.glutSolidSphere(radius, 20, 20)
            glPopMatrix()
            
            # Render action shape - EXACTLY as original but enhanced
            with self.data_lock:
                action = self.action
                
            shape, color = self.action_to_visual.get(action, ('sphere', (0.3, 0.3, 0.3, 0.7)))
            glColor4f(*color)
            glPushMatrix()
            glTranslatef(position[0] + 0.5, position[1] + 0.5, 1.0)
            if shape == 'cube':
                self.glutSolidCube(0.3)
            else:
                self.glutSolidSphere(0.3, 20, 20)
            glPopMatrix()
            
            # Render reward cubes - EXACTLY as original but enhanced
            with self.data_lock:
                reward = self.reward
                
            if reward >= 0:
                glColor4f(0.0, 1.0, 0.0, 0.8)
            else:
                glColor4f(1.0, 0.0, 0.0, 0.8)
            glPushMatrix()
            glTranslatef(position[0] + 0.5, position[1] + 0.5, 0.8)
            self.glutSolidCube(0.2)
            glPopMatrix()
            
            # Render progress bar - enhanced but EXACTLY same logic as original
            with self.data_lock:
                gestational_week = self.gestational_week
                
            glColor4f(0.0, 1.0, 0.0, 0.8)
            glBegin(GL_QUADS)
            progress = (gestational_week - 12) / (40 - 12)
            glVertex3f(-4, -4, 0)
            glVertex3f(-4 + 8 * progress, -4, 0)
            glVertex3f(-4 + 8 * progress, -3.8, 0)
            glVertex3f(-4, -3.8, 0)
            glEnd()
            
            pygame.display.flip()
            
        except Exception as e:
            print(f"Rendering error (continuing): {e}")
    
    def print_episode_summary(self):
        """EXACTLY as your original"""
        with self.data_lock:
            episode = self.episode
            total_reward = self.total_reward
            
        print(f"Episode {episode} Summary: Total Reward = {total_reward:.2f}")
    
    def demo_with_real_agent(self, model=None, model_type="Random"):
        """Demo with real agent - EXACTLY as your original but with enhancements"""
        try:
            from environment.custom_env import GDMEnvironment
            env = GDMEnvironment() if model else None
        except ImportError:
            env = None
            print("Warning: Could not import GDMEnvironment")
        
        clock = pygame.time.Clock()
        viz_data = []
        
        episodes = 10
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}: {model_type} Agent")
            self.reset_episode_stats(episode + 1)
            
            current_pos = (0, 0)
            gestational_week = 12
            obs = env.reset() if env else None
            self.update_state(current_pos, GDMRiskLevel.LOW, ActionType.ROUTINE_MONITORING, 0, gestational_week)
            
            for step in range(20):
                if not self.running:
                    break
                
                self.handle_events()
                
                # Use your original get_risk_level function or fallback - EXACT as original
                try:
                    risk_level = get_risk_level(gestational_week, None)
                except:
                    # Fallback risk level calculation - EXACT as original
                    if gestational_week < 20:
                        risk_level = GDMRiskLevel.LOW
                    elif gestational_week < 30:
                        risk_level = GDMRiskLevel.MODERATE
                    elif gestational_week < 35:
                        risk_level = GDMRiskLevel.HIGH
                    else:
                        risk_level = GDMRiskLevel.CRITICAL
                
                # Model prediction - EXACT as original
                if model:
                    if model_type == "REINFORCE":
                        obs_tensor = torch.FloatTensor(obs)
                        action_probs = model(obs_tensor)
                        action = torch.argmax(action_probs).item()
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                else:
                    action = np.random.choice(list(range(15)))
                
                action_type = ActionType(action)
                
                # Use your original reward calculation - EXACT as original
                try:
                    reward = env.calculate_reward(risk_level, action_type) if env else self.calculate_simple_reward(risk_level, action_type)
                except:
                    reward = self.calculate_simple_reward(risk_level, action_type)
                
                # Use your original position choosing - EXACT as original
                try:
                    from environment.utils import choose_next_position
                    current_pos = choose_next_position(current_pos, risk_level, action_type, step, gestational_week)
                except:
                    current_pos = self.simple_position_update(current_pos, risk_level, action_type, gestational_week)
                
                gestational_week += np.random.randint(1, 3)
                gestational_week = min(40, gestational_week)
                
                if env:
                    obs = np.array([current_pos[0], current_pos[1], gestational_week], dtype=np.float32)
                
                self.update_state(current_pos, risk_level, action_type, reward, gestational_week)
                viz_data.append({
                    "episode": episode + 1,
                    "step": step + 1,
                    "position": current_pos,
                    "risk_level": risk_level.name,
                    "action": action_type.name,
                    "reward": reward,
                    "gestational_week": gestational_week
                })
                
                print(f"  Step {step+1}: Week {gestational_week:.1f}, Pos: {current_pos}, "
                      f"Risk: {risk_level.name}, Action: {action_type.name}, Reward: {reward:.1f}")
                
                self.render()
                clock.tick(10)
                time.sleep(0.5)
                
                if current_pos == (7, 7) or gestational_week >= 40:
                    if current_pos == (7, 7):
                        print(f"  🎉 Patient reached optimal care path!")
                    else:
                        print(f"  👶 Pregnancy completed at week {gestational_week:.1f}")
                    break
            
            self.print_episode_summary()
            
            # Save log - EXACT as original
            try:
                import os
                os.makedirs("logs", exist_ok=True)
                pd.DataFrame(viz_data).to_csv(f"logs/viz_log_{model_type.lower()}_exp{episode + 1}.csv", index=False)
                print(f"Visualization log saved: logs/viz_log_{model_type.lower()}_exp{episode + 1}.csv")
            except:
                print("Could not save visualization log")
            
            time.sleep(3)
        
        pygame.quit()
    
    def calculate_simple_reward(self, risk_level, action):
        """Simple reward calculation fallback - EXACT as your original"""
        base_reward = 0
        
        if risk_level == GDMRiskLevel.LOW:
            if action == ActionType.ROUTINE_MONITORING:
                base_reward = 5
            elif action in [ActionType.DIETARY_COUNSELING, ActionType.EXERCISE_PROGRAM]:
                base_reward = 3
            else:
                base_reward = 1
        elif risk_level == GDMRiskLevel.MODERATE:
            if action in [ActionType.INCREASED_MONITORING, ActionType.GLUCOSE_TOLERANCE_TEST]:
                base_reward = 5
            else:
                base_reward = 2
        elif risk_level == GDMRiskLevel.HIGH:
            if action in [ActionType.CONTINUOUS_GLUCOSE_MONITORING, ActionType.SPECIALIST_REFERRAL]:
                base_reward = 6
            else:
                base_reward = 2
        elif risk_level == GDMRiskLevel.CRITICAL:
            if action == ActionType.IMMEDIATE_INTERVENTION:
                base_reward = 8
            else:
                base_reward = 3
        
        noise = np.random.normal(0, 0.5)
        return base_reward + noise
    
    def simple_position_update(self, current_pos, risk_level, action, gestational_week):
        """Simple position update fallback - EXACT as your original"""
        row, col = current_pos
        
        # Bias toward goal in later pregnancy
        if gestational_week > 30:
            if row < 7 and col < 7:
                return (row + 1, col + 1)
            elif row < 7:
                return (row + 1, col)
            elif col < 7:
                return (row, col + 1)
        
        # Random adjacent movement
        moves = [(row-1, col), (row+1, col), (row, col-1), (row, col+1), (row, col)]
        valid_moves = [(r, c) for r, c in moves if 0 <= r < 8 and 0 <= c < 8]
        
        return valid_moves[np.random.randint(len(valid_moves))] if valid_moves else current_pos
    
    def run_visualization(self, model=None, model_type="Random"):
        """Main visualization runner - EXACTLY as your original"""
        try:
            self.print_instructions()
            self.demo_with_real_agent(model, model_type)
        except Exception as e:
            print(f"Error in run_visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pygame.quit()

# Main execution functions - EXACTLY compatible with your original
def main():
    """Main function - drop-in replacement for your original"""
    print("🏥 GDM Care Journey - Enhanced 3D Visualization")
    
    mode = input("\n1. Interactive mode\n2. Medical simulation\nChoose (1-2): ").strip()
    
    viz = CleanGDMVisualization()
    
    if mode == "2":
        # Start visualization in background
        viz_thread = threading.Thread(target=viz.run_visualization)
        viz_thread.daemon = True
        viz_thread.start()
        
        time.sleep(2)  # Let visualization initialize
        # Run medical simulation
        viz.demo_with_real_agent()
    else:
        # Interactive mode
        viz.run_visualization()

# Additional helper functions for full compatibility with your original environment

def run_with_trained_model(model, model_type):
    """Run visualization with a trained model - EXACTLY as your original interface"""
    viz = CleanGDMVisualization()
    viz.run_visualization(model, model_type)

def run_interactive_demo():
    """Run interactive demo - EXACTLY as your original"""
    viz = CleanGDMVisualization()
    
    print("\n🎮 Interactive Demo Mode")
    print("Press SPACE to simulate steps, ESC to exit")
    
    # Start with initial state
    viz.update_state((0, 0), GDMRiskLevel.LOW, ActionType.ROUTINE_MONITORING, 0, 12)
    
    clock = pygame.time.Clock()
    step = 0
    
    while viz.running:
        viz.handle_events()
        
        # Check for space key press for manual simulation
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            # Simulate next step
            step += 1
            
            # Random state changes for demo
            import random
            current_pos = viz.position
            row, col = current_pos
            
            # Random adjacent move
            moves = [(row-1,col), (row+1,col), (row,col-1), (row,col+1)]
            valid_moves = [(r,c) for r,c in moves if 0 <= r < 8 and 0 <= c < 8]
            
            if valid_moves:
                new_pos = random.choice(valid_moves)
            else:
                new_pos = current_pos
            
            new_risk = random.choice(list(GDMRiskLevel))
            new_action = random.choice(list(ActionType))
            reward = random.uniform(-3, 5)
            new_week = min(40, viz.gestational_week + random.randint(1, 2))
            
            viz.update_state(new_pos, new_risk, new_action, reward, new_week)
            
            print(f"Step {step}: Week {new_week}, Pos {new_pos}, Risk: {new_risk.name}, Action: {new_action.name}, Reward: {reward:.1f}")
            
            time.sleep(0.2)  # Prevent too rapid simulation
        
        viz.render()
        clock.tick(60)
    
    pygame.quit()

# Utility functions for compatibility with your original codebase

def create_visualization_instance():
    """Create a visualization instance - for compatibility with your original code"""
    return CleanGDMVisualization()

def run_episode_visualization(viz, model=None, model_type="Random", episode_num=1):
    """Run a single episode visualization - for integration with your training loops"""
    print(f"\n🏥 Episode {episode_num}: {model_type} Agent Visualization")
    
    viz.reset_episode_stats(episode_num)
    
    # Initialize episode
    current_pos = (0, 0)
    gestational_week = 12
    step = 0
    
    # Try to use your environment
    try:
        from environment.custom_env import GDMEnvironment
        env = GDMEnvironment()
        obs = env.reset()
    except ImportError:
        env = None
        obs = None
        print("Warning: Could not import GDMEnvironment, using fallback")
    
    viz.update_state(current_pos, GDMRiskLevel.LOW, ActionType.ROUTINE_MONITORING, 0, gestational_week)
    
    episode_data = []
    
    for step in range(20):
        # Handle pygame events
        viz.handle_events()
        if not viz.running:
            break
        
        # Get risk level using your original function
        try:
            risk_level = get_risk_level(gestational_week, None)
        except:
            # Fallback - same logic as your original
            if gestational_week < 20:
                risk_level = GDMRiskLevel.LOW
            elif gestational_week < 30:
                risk_level = GDMRiskLevel.MODERATE
            elif gestational_week < 35:
                risk_level = GDMRiskLevel.HIGH
            else:
                risk_level = GDMRiskLevel.CRITICAL
        
        # Get action from model or random
        if model:
            if model_type == "REINFORCE":
                obs_tensor = torch.FloatTensor(obs) if obs is not None else torch.FloatTensor([current_pos[0], current_pos[1], gestational_week])
                action_probs = model(obs_tensor)
                action = torch.argmax(action_probs).item()
            else:
                if obs is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = np.random.choice(list(range(15)))
        else:
            action = np.random.choice(list(range(15)))
        
        action_type = ActionType(action)
        
        # Calculate reward using your original method
        try:
            reward = env.calculate_reward(risk_level, action_type) if env else viz.calculate_simple_reward(risk_level, action_type)
        except:
            reward = viz.calculate_simple_reward(risk_level, action_type)
        
        # Update position using your original method
        try:
            from environment.utils import choose_next_position
            current_pos = choose_next_position(current_pos, risk_level, action_type, step, gestational_week)
        except:
            current_pos = viz.simple_position_update(current_pos, risk_level, action_type, gestational_week)
        
        # Advance pregnancy
        gestational_week += np.random.randint(1, 3)
        gestational_week = min(40, gestational_week)
        
        # Update observation for next step
        if env:
            obs = np.array([current_pos[0], current_pos[1], gestational_week], dtype=np.float32)
        
        # Update visualization
        viz.update_state(current_pos, risk_level, action_type, reward, gestational_week)
        
        # Store episode data
        episode_data.append({
            "episode": episode_num,
            "step": step + 1,
            "position": current_pos,
            "risk_level": risk_level.name,
            "action": action_type.name,
            "reward": reward,
            "gestational_week": gestational_week
        })
        
        print(f"  Step {step+1}: Week {gestational_week:.1f}, Pos: {current_pos}, "
              f"Risk: {risk_level.name}, Action: {action_type.name}, Reward: {reward:.1f}")
        
        # Render
        viz.render()
        time.sleep(0.8)  # Slower for better observation
        
        # Check termination conditions
        if current_pos == (7, 7) or gestational_week >= 40:
            if current_pos == (7, 7):
                print(f"  🎉 Patient reached optimal care path!")
            else:
                print(f"  👶 Pregnancy completed at week {gestational_week:.1f}")
            break
    
    viz.print_episode_summary()
    return episode_data

# For backward compatibility with your existing code
ClearGDMVisualization = CleanGDMVisualization  # Alias for your original class name

# if __name__ == "__main__":
#     main()