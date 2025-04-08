import torch.optim.optimizer
from board_space import BoardSpace
from space_type import SpaceType
from monopoly_logger import get_monopoly_logger
from dqn import DQN, ReplayBuffer
import torch
import numpy as np
from constants import *
from typing import Union, List

class Agent:
    def __init__(self, money: int = 1500, logging_level: str = 'info',eps_start = 1,eps_end = .001,eps_decay = 50000, device = "cuda" if torch.cuda.is_available() else "cpu", hidden_layers:int = 3,hidden_layer_sizes:Union[List[int], int] =[200,200,200]):
        self.money = money
        self.logging_level = logging_level
        self.device = device
        assert hidden_layers == len(hidden_layer_sizes), "Please enter list of hidden layer sizes that equal amount of hidden layers."
        self.policy_net = DQN((BATCH_SIZE, 4,40), hidden_layers, hidden_layer_sizes, 40).to(device=self.device)
        self.target_net = DQN((BATCH_SIZE, 4,40), hidden_layers, hidden_layer_sizes, 40).to(device=self.device)
        self.target_net.eval()
        self.experience = ReplayBuffer(MAX_EXPERIENCE_LEN)
        self.logger = get_monopoly_logger(__name__, self.logging_level)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.optimizer = torch.optim.Adam(DQN.parameters(), .001)
        self.loss = torch.nn.MSELoss()

    def get_action(self, state: np.ndarray, episode_num:int):
        self.logger.info('Getting next agent action')
        # return random action right now but then implement Deep Q-learning
        r = np.max([(self.eps_decay - episode_num)/self.eps_decay, 0])
        eps = (self.epsmax - self.epsmin)*r + self.epsmin
        if np.random.uniform() < eps:
             return np.random.randint(0, 40)
        else:
            return self.policy_net(torch.tensor(state, dtype = torch.int64).to(device=self.device)).argmax(dim=1).item()
        
    def optimize(self):
        if len(self.experience) < 4*BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.experience.sample(BATCH_SIZE)
        states = states.to(device=self.device)
        actions = actions.to(device=self.device)
        rewards = rewards.to(device=self.device)
        next_states = next_states.to(device=self.device)
        dones = dones.to(device=self.device)
        self.optimizer._zero_grad()
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_vals = torch.max(self.target_net(next_states), 1)[0]
        target_q_values = rewards + GAMMA * next_q_vals
        loss = self.loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


