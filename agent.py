from board_space import BoardSpace
from space_type import SpaceType
from monopoly_logger import get_monopoly_logger

import numpy as np


class Agent:
    def __init__(self, money: int = 1500, logging_level: str = 'info'):
        self.money = money
        self.logging_level = logging_level

        self.logger = get_monopoly_logger(__name__, self.logging_level)

    def get_action(self, state: np.ndarray):
        self.logger.info('Getting next agent action')
        # return random action right now but then implement Deep Q-learning
        return np.random.randint(0, 40)
        # or continue trying until space is property type
