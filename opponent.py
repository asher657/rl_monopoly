import numpy as np

from monopoly_logger import get_monopoly_logger


class Opponent:
    """
    Represents an opponent in a Monopoly-like game.

    Attributes:
        curr_position (int): The current position of the opponent on the board.
        money (int): The amount of money the opponent has.
        logging_level (str): The logging level ('info', 'debug', or 'none').
        logger (logging.Logger): Logger instance for logging opponent actions.
    """
    def __init__(self, curr_position: int = 0, money: int = 1500, logging_level: str = 'info'):
        self.curr_position = curr_position
        self.money = money
        self.logging_level = logging_level

        self.logger = get_monopoly_logger(__name__, self.logging_level)

    def get_action(self) -> int:
        """
        Simulates the opponent rolling two dice and returns the sum.

        Returns:
            int: The total value of the dice roll.
        """
        self.logger.info('Getting opponent next action')
        roll_1 = np.random.randint(1, 7)
        roll_2 = np.random.randint(1, 7)
        self.logger.debug(f'Opponent rolled {roll_1} and {roll_2} with total {roll_1 + roll_2}')

        return roll_1 + roll_2
