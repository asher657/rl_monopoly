import torch.optim.optimizer

from agents.agent import Agent
from utils.monopoly_logger import get_monopoly_logger
from agents.dqn import DQN, ReplayBuffer
import torch
import numpy as np
from utils.constants import *
from typing import Union, List


class DqnAgent(Agent):
    def __init__(self,
                 money: int = AGENT_STARTING_MONEY,
                 logging_level: str = 'info',
                 eps_start=1,
                 eps_end=.001,
                 eps_decay=100,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 256,
                 gamma: float = 0.9,
                 max_experience_len: int = 16384,
                 lr: float = 0.001,
                 update_target_net_freq: int = 50,
                 hidden_layer_sizes: Union[List[int], int] = [200, 200, 200]):
        super().__init__(money, logging_level)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_experience_len = max_experience_len
        self.lr = lr
        self.update_target_net_freq = update_target_net_freq

        self.eps = eps_start
        self.experience = ReplayBuffer(self.max_experience_len)
        self.logger = get_monopoly_logger(__name__, self.logging_level)

        self.loss = torch.nn.MSELoss()
        self.policy_net = DQN((self.batch_size, 5, 40), hidden_layer_sizes, 40).to(device=self.device)
        self.target_net = DQN((self.batch_size, 5, 40), hidden_layer_sizes, 40).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), self.lr)
        self.target_net.eval()

    def get_action(self, state: np.ndarray):
        self.logger.info('Getting next agent action')
        if np.random.uniform() < self.eps:
            self.logger.debug("Chose random action")
            return np.random.randint(0, 40)
        else:
            self.logger.debug("Chose calculated action")
            output = self.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device=self.device))
            action = torch.argmax(output).item()
            return action

    def update_epsilon(self, episode_num: int):
        r = np.max([(self.eps_decay - episode_num) / self.eps_decay, 0])
        self.eps = (self.eps_start - self.eps_end) * r + self.eps_end
        if self.eps == self.eps_end:
            self.logger.info('=== Epsilon reached the minimum ===')

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def optimize(self, episode):
        if len(self.experience) < BATCH_MULTIPLIER * self.batch_size:
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
        target_q_values = rewards + ~dones * self.gamma * next_q_vals

        loss = self.loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        if episode % self.update_target_net_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
