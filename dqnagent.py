import torch.optim.optimizer
from monopoly_logger import get_monopoly_logger
from dqn import DQN, ReplayBuffer
import torch
import numpy as np
from constants import *
from typing import Union, List


class DqnAgent:
    def __init__(self,
                 money: int = AGENT_STARTING_MONEY,
                 logging_level: str = 'info',
                 eps_start=1,
                 eps_end=.001,
                 eps_decay=50000,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 256,
                 gamma: float = 0.9,
                 max_experience_len: int = 5120,
                 hidden_layers: int = 3,
                 lr: float = 0.001,
                 hidden_layer_sizes: Union[List[int], int] = [200, 200, 200]):
        self.money = money
        self.logging_level = logging_level
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_experience_len = max_experience_len
        assert hidden_layers == len(
            hidden_layer_sizes), "Please enter list of hidden layer sizes that equal amount of hidden layers."
        self.lr = lr

        self.experience = ReplayBuffer(self.max_experience_len)
        self.logger = get_monopoly_logger(__name__, self.logging_level)

        self.loss = torch.nn.MSELoss()
        self.policy_net = DQN((self.batch_size, 4, 40), hidden_layers, hidden_layer_sizes, 40).to(device=self.device)
        self.target_net = DQN((self.batch_size, 4, 40), hidden_layers, hidden_layer_sizes, 40).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.lr)
        self.target_net.eval()

    def get_action(self, state: np.ndarray, episode_num: int):
        self.logger.info('Getting next agent action')
        r = np.max([(self.eps_decay - episode_num) / self.eps_decay, 0])
        eps = (self.eps_start - self.eps_end) * r + self.eps_end
        if np.random.uniform() < eps:
            return np.random.randint(0, 40)
        else:
            return self.policy_net(torch.tensor(state, dtype=torch.int64).to(device=self.device)).argmax(dim=1).item()

    def optimize(self):
        if len(self.experience) < 4 * self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.experience.sample(self.batch_size)
        states = states.to(device=self.device)
        actions = actions.to(device=self.device)
        rewards = rewards.to(device=self.device)
        next_states = next_states.to(device=self.device)
        dones = dones.to(device=self.device)
        self.optimizer.zero_grad()
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_vals = torch.max(self.target_net(next_states), 1)[0]
        target_q_values = rewards + self.gamma * next_q_vals
        loss = self.loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
