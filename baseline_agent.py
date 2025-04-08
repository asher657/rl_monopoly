from constants import *
from agent import Agent
from monopoly_logger import get_monopoly_logger

import numpy as np


class BaselineAgent(Agent):
    """
    A baseline Monopoly agent that makes purchasing decisions based on opponent position.

    This agent determines which property to buy based on the opponent's current position,
    aiming to purchase properties 7 spaces ahead of the opponent. If the desired property
    is not available, it selects the closest buyable property.

    Attributes:
        logger (logging.Logger): Logger instance for the agent.

    Args:
        money (int): The starting money for the agent. Defaults to 1500.
        logging_level (str): The logging level for the agent. Defaults to 'info'.
    """
    def __init__(self, money: int = AGENT_STARTING_MONEY, logging_level: str = 'info'):
        super().__init__(money, logging_level)
        self.logger = get_monopoly_logger(__name__, self.logging_level)

    def get_action(self, state: np.ndarray):
        """
        Determines the next property purchase action based on the current game state.

        The agent identifies the opponent's position and attempts to buy the property
        located 7 spaces ahead. If that property is unavailable, it buys the nearest
        available property.

        Args:
            state (np.ndarray): The current game state, including player money and opponent position.

        Returns:
            int: The index of the property the agent decides to purchase.
        """
        self.logger.info('Getting next agent action')
        opponent_position = np.where(state[OPPONENT_SPACE_INDEX] == 1)[0][0]
        buyable_positions = np.where(state[PLAYER_MONEY_INDEX] == 1)[0]
        buy_position = opponent_position + 7
        if buy_position >= 40:
            # pass go
            buy_position = buy_position % 40

        self.logger.debug(f'Opponent on position: {opponent_position}, attempting to buy: {buy_position}')

        if buy_position in buyable_positions:
            self.logger.debug(f'Agent is able to buy house on: {buy_position}')
            return buy_position
        else:
            nearest_buy_position = buyable_positions[np.abs(buyable_positions - buy_position).argmin()]
            self.logger.debug(f'Agent is unable to buy house on: {buy_position}. Buying: {nearest_buy_position}')
            return nearest_buy_position

