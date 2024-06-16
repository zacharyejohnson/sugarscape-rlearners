import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque


class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions, learning_rate, discount_rate, n_layers, layer_size):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_inputs, layer_size))  # Use n_inputs as the input size
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(layer_size, layer_size))
        self.layers.append(nn.Linear(layer_size, n_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def predict_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()

    def update(self, states, actions, rewards, next_states, discount_rate, n):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.forward(states)
        next_q_values = self.forward(next_states)

        targets = q_values.clone().detach()  
        for i in range(len(states)):
            target = rewards[i] + discount_rate * next_q_values[i].max()
            targets[i, actions[i]] = target

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, transitions, discount_rate):
        states, actions, rewards, next_states = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.forward(states)
        next_q_values = self.forward(next_states)

        targets = rewards + discount_rate * next_q_values.max(dim=1)[0]
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values, targets)
        return loss



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append(Transition(state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def last(self):
        return self.memory.pop()
    
    def get_all_states(self):
        return np.array([transition.state for transition in self.memory])
