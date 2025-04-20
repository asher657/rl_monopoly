import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Union, List, Tuple


class DQN(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 hidden_layer_sizes: Union[List[int], int],
                 n_actions: int):
        super(DQN, self).__init__()
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.multiple_layers = True if self.num_hidden_layers > 1 else False
        b, l, w = input_shape
        self.input_layers = nn.Linear(l * w, hidden_layer_sizes[0])

        if self.multiple_layers:
            self.hidden_layers = nn.ModuleList(
                [nn.Sequential(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]), nn.Dropout1d(.5)) for i in range(self.num_hidden_layers - 1)])

        self.output_layer = nn.Linear(hidden_layer_sizes[-1], n_actions)

    def forward(self, x: torch.tensor):
        assert len(x.shape) == 3, "Please have at least 1 in the batch dimension."
        x = x.flatten(start_dim=1, end_dim=2)
        x = F.relu(self.input_layers(x))
        if self.multiple_layers:
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
        output = self.output_layer(x)
        return output


class ReplayBuffer:
    def __init__(self, max_length):
        self.experience = deque(maxlen=max_length)

    def push(self, transition):
        self.experience.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.experience, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.experience)
