import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque


device = torch.device("cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions, learning_rate, discount_rate, n_layers, layer_size):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_inputs, layer_size))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(layer_size, layer_size))
        self.layers.append(nn.Linear(layer_size, n_actions))
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

    def forward(self, state):
        x = state.to(device)
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x))  # Leaky ReLU allows for small negative values
        x = self.layers[-1](x)
        x = F.softplus(x)  # Ensures positive output
        return x

    def predict_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.forward(state)
            return torch.argmax(q_values).item()

    def update(self, states, actions, rewards, next_states):
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        q_values = self.forward(states)
        next_q_values = self.forward(next_states)

        targets = q_values.clone().detach()
        for i in range(len(states)):
            target = rewards[i] + next_q_values[i].max()
            targets[i, actions[i]] = target

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, transitions, discount_rate):
        states, actions, rewards, next_states = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        q_values = self.forward(states)
        next_q_values = self.forward(next_states)

        targets = rewards + discount_rate * next_q_values.max(dim=1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values, targets)
        return loss

class DoubleDQN(nn.Module):
    def __init__(self, n_inputs, n_actions, learning_rate, discount_rate, n_layers, layer_size):
        super(DoubleDQN, self).__init__()
        self.online_network = DQN(n_inputs, n_actions, learning_rate, discount_rate, n_layers, layer_size)
        self.target_network = DQN(n_inputs, n_actions, learning_rate, discount_rate, n_layers, layer_size)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def forward(self, state):
        return self.online_network.forward(state)

    def compute_loss(self, transitions, discount_rate):
        states, actions, rewards, next_states = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

        current_q_values = self.online_network.forward(states).gather(1, actions.unsqueeze(1))
        next_actions = self.online_network.forward(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_network.forward(next_states).gather(1, next_actions).squeeze(1)
        expected_q_values = rewards + (discount_rate * next_q_values)

        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        return loss

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class NStepReplayMemory:
    def __init__(self, capacity, n_step, gamma):
        self.memory = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        R = sum([self.gamma**i * t[2] for i, t in enumerate(self.n_step_buffer)])
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        self.memory.append((state, action, R, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class EpsilonGreedy:
    def __init__(self, start=1.0, end=0.01, decay=0.995):
        self.start, self.end, self.decay = start, end, decay
        self.epsilon = self.start

    def get_epsilon(self):
        return max(self.epsilon, self.end)

    def decay_epsilon(self):
        self.epsilon *= self.decay

class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0
        self.std = 0
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.std += delta * delta2
        if self.count > 1:
            return (reward - self.mean) / (np.sqrt(self.std / (self.count - 1)) + self.epsilon)
        return 0

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        else:
            self.buffer[self.position] = (state, action, reward, next_state)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return [], [], []

        if self.size < batch_size:
            batch_size = self.size

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        weights = (self.size * probs[indices]) ** (-self.beta)
        if weights.size > 0:
            weights /= weights.max()
        self.beta = np.min([1., self.beta + self.beta_increment])

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def last(self):
        return self.buffer[-1] if self.buffer else None

    def get_all_states(self):
        return np.array([transition[0] for transition in self.buffer])

    def __len__(self):
        return self.size