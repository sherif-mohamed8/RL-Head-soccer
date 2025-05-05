import numpy as np
import random
import pickle
from settings import *

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3, exploration_decay=0.9999):
        """
        Initialize Q-Learning Agent
        
        Args:
            learning_rate: Alpha, learning rate
            discount_factor: Gamma, future reward discount factor
            exploration_rate: Epsilon, probability of random action
            exploration_decay: Rate at which exploration decreases
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = 0.05
        
        # State discretization: divide field into grid cells
        self.grid_size_x = 8
        self.grid_size_y = 5
        
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_count = 4
        
        # Initialize Q-table as dictionary
        self.q_table = {}
        
        # Training metrics
        self.episode_count = 0
        self.total_reward = 0
        self.rewards_history = []
        
    def discretize_state(self, ai_pos, ball_pos, player_pos):
        """Convert continuous positions to discrete state"""
        # Convert positions to grid cells
        ai_x = int(ai_pos[0] / WIDTH * self.grid_size_x)
        ai_y = int(ai_pos[1] / HEIGHT * self.grid_size_y)
        ball_x = int(ball_pos[0] / WIDTH * self.grid_size_x)
        ball_y = int(ball_pos[1] / HEIGHT * self.grid_size_y)
        player_x = int(player_pos[0] / WIDTH * self.grid_size_x)
        player_y = int(player_pos[1] / HEIGHT * self.grid_size_y)
        
        # Ensure values are within bounds
        ai_x = max(0, min(ai_x, self.grid_size_x - 1))
        ai_y = max(0, min(ai_y, self.grid_size_y - 1))
        ball_x = max(0, min(ball_x, self.grid_size_x - 1))
        ball_y = max(0, min(ball_y, self.grid_size_y - 1))
        player_x = max(0, min(player_x, self.grid_size_x - 1))
        player_y = max(0, min(player_y, self.grid_size_y - 1))
        
        # Calculate relative positions (important for generalization)
        ball_rel_x = ball_x - ai_x
        ball_rel_y = ball_y - ai_y
        
        # Return as hashable state tuple
        return (ai_x, ai_y, ball_rel_x, ball_rel_y, ball_x, ball_y)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        # Explore: choose random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_count - 1)
        
        # Exploit: choose best action
        return self.get_best_action(state)
    
    def get_best_action(self, state):
        """Get best action for state based on Q-values"""
        if state not in self.q_table:
            # Initialize new state with zeros
            self.q_table[state] = np.zeros(self.action_count)
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using the Q-learning formula"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_count)
            
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_count)
            
        # Q-Learning formula: Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        best_next_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - current_q
        )
        
        # Update metrics
        self.total_reward += reward
    
    def end_episode(self):
        """Call at end of episode to update parameters"""
        self.rewards_history.append(self.total_reward)
        self.total_reward = 0
        self.episode_count += 1
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
    
    def save(self, filename='q_table.pkl'):
        """Save Q-table to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
    
    def load(self, filename='q_table.pkl'):
        """Load Q-table from file"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False

# Q-learning movement function to replace ai_move
def q_move(player, ball, opponent, agent, training=True):
    """
    Move the AI player using Q-learning
    
    Args:
        player: AI player object
        ball: Ball object
        opponent: Human player object
        agent: QLearningAgent instance
        training: Whether to train or just use policy
    """
    # Get state
    state = agent.discretize_state(player.pos, ball.pos, opponent.pos)
    
    # Choose action (explore/exploit)
    if training:
        action = agent.choose_action(state)
    else:
        action = agent.get_best_action(state)
    
    # Store player position before moving
    prev_pos = player.pos.copy()
    prev_ball_pos = ball.pos.copy()
    ball_dist_before = np.linalg.norm(player.pos - ball.pos)
    
    # Execute action based on action index
    speed = PLAYER_SPEED * 0.7  # Same speed reduction as original AI
    
    if action == 0:  # UP
        player.pos[1] -= speed
    elif action == 1:  # DOWN
        player.pos[1] += speed
    elif action == 2:  # LEFT
        player.pos[0] -= speed
    elif action == 3:  # RIGHT
        player.pos[0] += speed
    
    # Ensure player stays in bounds
    player._clamp(ball.bounds)
    
    if training:
        # Calculate reward
        reward = calculate_reward(player, ball, prev_pos, prev_ball_pos, ball_dist_before)
        
        # Get new state
        next_state = agent.discretize_state(player.pos, ball.pos, opponent.pos)
        
        # Update Q-values
        agent.update_q_value(state, action, reward, next_state)

def calculate_reward(player, ball, prev_pos, prev_ball_pos, ball_dist_before):
    """Calculate reward based on game state changes"""
    reward = 0
    
    # Distance to ball
    ball_dist_after = np.linalg.norm(player.pos - ball.pos)
    
    # Reward for moving closer to ball
    if ball_dist_after < ball_dist_before:
        reward += 1
    else:
        reward -= 0.5
        
    # Check if ball moved (possible hit)
    ball_movement = np.linalg.norm(ball.pos - prev_ball_pos)
    if ball_movement > 1.0:
        # Ball was hit
        reward += 5
        
        # Extra reward if ball moves toward opponent's goal (left side)
        if ball.vel[0] < 0:
            reward += 3
            
    # Penalty for being too close to own goal (right side)
    if player.pos[0] > WIDTH * 0.75:
        reward -= 1
        
    # Penalty for being too far from ball
    if ball_dist_after > WIDTH / 3:
        reward -= 0.5
            
    return reward