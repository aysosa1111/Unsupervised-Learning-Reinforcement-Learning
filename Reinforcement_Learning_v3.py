# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:54:53 2024

@author: Owner
"""

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import warnings

# Suppress Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()  # ReLU activation is used for non-linearity without affecting the scale of input values.
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size, learning_rate, gamma, buffer_size, batch_size, epsilon_start, epsilon_end, epsilon_decay, target_update):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.policy_net = DQN(state_size, action_size, hidden_size)
        self.target_net = DQN(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.target_update = target_update
        self.steps_done = 0
        
    def select_action(self, state):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()
        
    def push_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Current Q Values
        current_q = self.policy_net(state).gather(1, action)
        
        # Next Q Values
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * max_next_q
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training the DQN
def train_dqn(env, agent, num_episodes, max_steps):
    rewards = []
    steps_per_episode = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=SEED)  # Reset and get initial state
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)  # Handle truncated episodes
            agent.push_memory(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        rewards.append(total_reward)
        steps_per_episode.append(step)
        
        # Update the target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
    return rewards, steps_per_episode

# Plot Metrics
def plot_metrics(rewards, steps, window=100):
    plt.figure(figsize=(12,5))
    # Plot Rewards
    plt.subplot(1,2,1)
    plt.plot(rewards, label='Reward per Episode')
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Cumulative Rewards Over Episodes')
    plt.legend()
    
    # Plot Steps
    plt.subplot(1,2,2)
    plt.plot(steps, label='Steps per Episode')
    moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(steps)), moving_avg_steps, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Length Over Episodes')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Evaluate the Agent
def evaluate_agent(env, agent, num_episodes=10, max_steps=1000, render=False):
    total_rewards = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            if render:
                env.render()
            if done or truncated:
                break
        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode}: Reward: {total_reward:.2f}")
    env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} Evaluation Episodes: {avg_reward:.2f}")

# Main Execution
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    
    # Hyperparameters
    state_size = 8
    action_size = 4
    hidden_size = 128
    batch_size = 64
    learning_rate = 1e-3
    gamma = 0.99
    num_episodes = 1000
    max_steps = 1000
    buffer_size = 100000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update = 10
    
    agent = DQNAgent(state_size, action_size, hidden_size, learning_rate, gamma, buffer_size, batch_size, epsilon_start, epsilon_end, epsilon_decay, target_update)
    
    rewards, steps = train_dqn(env, agent, num_episodes, max_steps)
    plot_metrics(rewards, steps)
    torch.save(agent.policy_net.state_dict(), "dqn_lunarlander.pth")
    evaluate_agent(env, agent, num_episodes=10, max_steps=1000, render=False)
