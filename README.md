# Deep Q-Network (DQN) for LunarLander-v2

## Project Overview
This project implements a Deep Q-Network (DQN) reinforcement learning agent to solve the OpenAI Gym LunarLander-v2 environment. The goal is to land a lunar lander safely on the landing pad using deep neural networks and Q-learning techniques.

## Features
- Deep Q-Network (DQN) implementation
- Replay Buffer for experience replay
- Epsilon-greedy exploration strategy
- Target network for stable learning
- Visualization of training metrics
- Model saving and evaluation

## Prerequisites
- Python 3.8+
- Libraries:
  - gym
  - numpy
  - torch
  - matplotlib

## Installation
1. Clone the repository
2. Install required dependencies:
```bash
pip install gym numpy torch matplotlib
```

## Hyperparameters
- **State Size**: 8 dimensions
- **Action Size**: 4 actions
- **Hidden Layer Size**: 128 neurons
- **Learning Rate**: 1e-3
- **Discount Factor (γ)**: 0.99
- **Batch Size**: 64
- **Replay Buffer Size**: 100,000
- **Epsilon Decay**: 0.995
- **Target Network Update**: Every 10 episodes

## Training
The training process involves:
- Exploration using epsilon-greedy policy
- Experience replay for learning stability
- Target network for Q-value estimation
- Tracking rewards and episode steps

### Training Metrics
The script generates plots for:
- Cumulative Rewards
- Episode Length
- Moving Averages

## Saved Artifacts
- Trained Model: `dqn_lunarlander.pth`
- Training Plots: Generated during runtime

## Usage
```bash
python lunar_lander_dqn.py
```

## Results
The agent learns to land the lunar lander with increasing efficiency over multiple episodes, demonstrating the effectiveness of the DQN approach.

## References
- [OpenAI Gym](https://gym.openai.com/)
- [Deep Q-Learning Paper](https://www.nature.com/articles/nature14236)

