from board_space import BoardSpace
from space_type import SpaceType
import numpy as np



class Agent:
    def __init__(self, money: int = 1500):
        self.money = money

    def get_action(self, state):
        """"
        bought_houses: binary of shape [40]
        opponent_pos: one-hot of shape [40]
        property_rents: static of ints shape [40] - all the rents for one house
        opponent_money: binary of shape [40] - which rents the opponent can afford
        agent_money: binary of shape [40] - which houses the agent can afford
        total: shape [1, 200]
        Deep Q-Learning is only option
        """
        # return random action right now but then implement Deep Q-learning
        return np.random.randint(0, 40)
        # or continue trying until space is property type
