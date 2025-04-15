import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque

class EmergencyQNetwork(nn.Module):
    """Neural network for approximating Q-values in the emergency rerouting scenario."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(EmergencyQNetwork, self).__init__()
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
        
        # Pad states and next_states to ensure uniform shape if needed
        max_dim = max(len(s) for s in states)
        states_padded = []
        next_states_padded = []
        
        for s in states:
            if len(s) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(s)] = s
                states_padded.append(padded)
            else:
                states_padded.append(s)
        
        for s in next_states:
            if len(s) < max_dim:
                padded = np.zeros(max_dim, dtype=np.float32)
                padded[:len(s)] = s
                next_states_padded.append(padded)
            else:
                next_states_padded.append(s)
        
        return (np.array(states_padded),
                np.array(actions),
                np.array(rewards),
                np.array(next_states_padded),
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class EmergencyDQNAgent:
    def __init__(self, state_dim, action_dim, 
                 lr=0.0001,              # Lower learning rate
                 gamma=0.99,             # Discount factor
                 epsilon=1.0,            # Initial exploration rate
                 epsilon_min=0.05,       # Minimum exploration rate
                 epsilon_decay=0.9995,   # Decay rate for exploration
                 buffer_size=50000,      # Replay buffer size
                 batch_size=64,          # Batch size for updates
                 update_frequency=10,    # How frequently to update the network
                 target_update=500):     # How often to update the target network
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_counter = 0
        self.update_frequency = update_frequency
        self.target_update = target_update
        
        # Initialize the emergency Q-network and target network
        self.q_network = EmergencyQNetwork(state_dim, action_dim, hidden_dim=128)
        self.target_network = EmergencyQNetwork(state_dim, action_dim, hidden_dim=128)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Use Adam optimizer with weight decay
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Metrics for monitoring
        self.losses = []
        self.avg_q_values = []
        self.exploration_rate = []
    
    def select_action(self, state):
        """Select an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        """Perform a training update on the Q-network."""
        # Convert state and next_state to numpy arrays if needed
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32)
        else:
            state = np.array(state, dtype=np.float32)
        if isinstance(next_state, np.ndarray):
            next_state = next_state.astype(np.float32)
        else:
            next_state = np.array(next_state, dtype=np.float32)
        
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Wait until we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        self.update_counter += 1
        
        # Only update every 'update_frequency' steps
        if self.update_counter % self.update_frequency != 0:
            return 0
        
        try:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            # Ensure proper type conversion
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            
            # Convert to PyTorch tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            
            # Compute current Q-values for the batch
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values using the target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss using smooth L1 loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Periodically update the target network
            if self.update_counter % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Record metrics
            self.losses.append(loss.item())
            self.avg_q_values.append(current_q_values.mean().item())
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.exploration_rate.append(self.epsilon)
            
            print(f"Emergency network updated: Loss = {loss.item():.6f}, Mean Q = {current_q_values.mean().item():.6f}")
            
            return loss.item()
        
        except Exception as e:
            print(f"Error in emergency network update: {e}")
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            return 0
    
    def save(self, filepath):
        """Save the emergency DQN model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath, weights_only=False):
        """Load the DQN model."""
        if not os.path.exists(filepath):
            return
            
        checkpoint = torch.load(filepath, weights_only=weights_only)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
