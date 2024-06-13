import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque


class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions, learning_rate, discount_rate):
        super(DQN, self).__init__()
        self.fc = nn.Linear(n_inputs, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

    def forward(self, state):
        return self.fc(state)

    def predict_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, discount_rate):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.forward(state)
        next_q_values = self.forward(next_state)

        target = reward + discount_rate * torch.max(next_q_values).item()
        loss = self.loss_fn(q_values[0, action], torch.tensor([target]))

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
