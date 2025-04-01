import torch.optim.optimizer
from board_space import BoardSpace
from space_type import SpaceType
from monopoly_logger import get_monopoly_logger
from dqn import DQN, ReplayBuffer
import torch
import numpy as np
from constants import *


class Agent:
    def __init__(self, money: int = 1500, logging_level: str = 'info', episode_number: int = 0):
        self.money = money
        self.logging_level = logging_level
        self.policy_net = DQN((BATCH_SIZE, 4,40), 3, [200,200,200], 40)
        self.target_net = None
        self.experience = ReplayBuffer(MAX_EXPERIENCE_LEN)
        self.logger = get_monopoly_logger(__name__, self.logging_level)
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.episode_number = episode_number
        self.optimizer = torch.optim.Adam(DQN.parameters(), .001)

    def get_action(self, state: np.ndarray):
        self.logger.info('Getting next agent action')
        # return random action right now but then implement Deep Q-learning
        r = np.max([(self.eps_decay - self.episode_num)/self.eps_decay, 0])
        eps = (self.epsmax - self.epsmin)*r + self.epsmin
        if np.random.uniform() < eps:
             return np.random.randint(0, 40)
        else:
            return np.argmax(self.policy_net(torch.tensor(state, dtype = torch.int64)))
        
    def optimize(self):
        pass
