import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque

class QNetwork(nn.Module):
    """Neural network for approximating Q-values."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[idx] for idx in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Make sure all states have the same shape by padding if necessary
        max_dim = max(len(s) for s in states)
        states_padded = []
        next_states_padded = []
        
        for s in states:
            if len(s) < max_dim:
                # Pad with zeros to match the largest dimension
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(s)] = s
                states_padded.append(padded)
            else:
                states_padded.append(s)
        
        for s in next_states:
            if len(s) < max_dim:
                # Pad with zeros to match the largest dimension
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(s)] = s
                next_states_padded.append(padded)
            else:
                next_states_padded.append(s)
        
        return np.array(states_padded), np.array(actions), np.array(rewards), np.array(next_states_padded), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN agent using function approximation instead of tabular Q-values."""
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, 
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64, update_frequency=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_counter = 0
        self.update_frequency = update_frequency
        
        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Statistics
        self.losses = []
        self.exploration_rate = []
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() < self.epsilon:
            # Exploration
            return np.random.randint(self.action_dim)
        else:
            # Exploitation
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        """Store transition and perform learning updates."""
        # Store transition in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Increment update counter
        self.update_counter += 1
        
        # Only update every update_frequency steps
        if self.update_counter % self.update_frequency != 0:
            return 0
            
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Compute current Q-values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.exploration_rate.append(self.epsilon)
        
        # Periodically update target network
        if self.update_counter % (self.update_frequency * 10) == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()
    
    def save(self, filepath):
        """Save the DQN model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load the DQN model."""
        if not os.path.exists(filepath):
            return
            
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']