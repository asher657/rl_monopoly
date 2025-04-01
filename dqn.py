import torch
import torch.nn as nn
import torch.functional as FF
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Union, List, Tuple



class DQN(nn.Module):
    def __init__(self,input_shape: Tuple[int],hidden_layers:int,hidden_layer_sizes:Union[List[int], int], n_actions: int):
        super(DQN, self).__init__()
        self.hidden_layers = hidden_layers
        self.multiple_layers = True if hidden_layers > 1 else False
        b,l,w = input_shape
        if self.multiple_layers:
            self.input_layers = nn.Linear(l*w, hidden_layer_sizes[0]),
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]) for i in range(hidden_layers - 1)])
            self.output_layer = nn.Linear(hidden_layer_sizes[-1], n_actions)
        else:
            self.input_layers = nn.Linear(l*w, hidden_layer_sizes)
            self.output_layer = nn.Linear(hidden_layer_sizes, n_actions)
    def forward(self,x):
        x = nn.ReLU(self.input_layers(x))
        if self.multiple_layers:
            for layer in self.hidden_layers:
                x = nn.RelU(layer(x))
            output = self.output_layer(x)
        else:
            x = nn.RelU(self.hidden_layers(x))
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
            torch.tensor(np.array(states), dtype=torch.int64),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.int64),
            torch.tensor(np.array(next_states), dtype=torch.int64),
            torch.tensor(dones, dtype=torch.bool)
        )
    def __len__(self):
        return len(self.experience)
