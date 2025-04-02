import json
import os.path
from typing import Dict
import numpy as np

from board_space import BoardSpace
from constants import *
from opponent import Opponent
from space_type import SpaceType
from monopoly_logger import get_monopoly_logger


class Board:
    """
    Represents the game board, including the player's opponent, board spaces, and game state.

    Attributes:
        logging_level (str): The logging level ('info', 'debug', or 'none').
        default_cost (int): The default cost associated with properties where a house cannot be bought.
        logger (logging.Logger): Logger instance for logging board actions.
        opponent (Opponent): The opponent in the game.
        board_positions (Dict[int, BoardSpace]): A dictionary mapping board positions to BoardSpace objects.
        state (np.ndarray): A 2D numpy array representing the game state, with various indices for player state, houses, etc.
    """
    def __init__(self, default_cost: int = -10, logging_level: str = 'info', max_steps: int = 5000):
        self.logging_level = logging_level
        self.default_cost = default_cost
        self.max_steps = max_steps
        self.logger = get_monopoly_logger(__name__, logging_level)

        self.opponent = Opponent(logging_level=self.logging_level)
        self.board_positions = self._load_positions()
        self.state = np.zeros(shape=(5, 40))

        self.successes = []
        self.rewards = []
        self.agent_monies = []
        self.opponent_monies = []

        self.reset()

    def _load_positions(self) -> Dict[int, BoardSpace]:
        """
        Loads the board positions from a JSON file and returns them as a dictionary.

        Returns:
            Dict[int, BoardSpace]: A dictionary where the key is the position index and the value is a BoardSpace object.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'positions.json')
        self.logger.debug(f'Loading positions file from {file_path}')

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        return {int(key): BoardSpace(**value) for key, value in json_data.items()}

    def reset(self):
        """
        Resets the board to its initial state, including resetting the opponent's position and updating rent values.
        """
        self.logger.debug('Resetting board')
        self.opponent = Opponent(logging_level=self.logging_level)
        for idx, space in self.board_positions.items():
            if space.space_type == SpaceType.PROPERTY:
                self.state[RENT_INDEX, idx] = space.rent[INITIAL_NUM_HOUSES]
        self.state[OPPONENT_SPACE_INDEX, self.opponent.curr_position] = 1
        self.update_affordability_states(AGENT_STARTING_MONEY)

    def update_affordability_states(self, agent_money: int):
        """
        Updates the affordability states based on the agent's and opponent's money.

        Args:
            agent_money (int): The amount of money the agent currently has.
        """
        self.logger.info('Updating affordability states')
        self.logger.debug(f'Agent money: {agent_money} and opponent money: {self.opponent.money}')
        for idx, space in self.board_positions.items():
            if space.can_build_house():
                if agent_money > space.rent[INITIAL_NUM_HOUSES]:
                    # self.logger.debug(f'Agent money greater than space {idx} rent')
                    self.state[PLAYER_MONEY_INDEX, idx] = 1
                else:
                    # self.logger.debug(f'Agent money less than space {idx} rent')
                    self.state[PLAYER_MONEY_INDEX, idx] = 0
                if self.opponent.money > space.rent[INITIAL_NUM_HOUSES]:
                    # self.logger.debug(f'Opponent money greater than space {idx} rent')
                    self.state[OPPONENT_MONEY_INDEX, idx] = 1
                else:
                    # self.logger.debug(f'Opponent money less than space {idx} rent')
                    self.state[OPPONENT_MONEY_INDEX, idx] = 0

    def update_state_space(self, opp_previous_pos, next_opponent_position: int, house_location: int, agent_money: int):
        """
        Updates the state space after an opponent moves, taking into account the current house location and the agent's money.

        Args:
            opp_previous_pos (int): The opponent's previous position.
            next_opponent_position (int): The opponent's next position.
            house_location (int): The location of the house being considered.
            agent_money (int): The amount of money the agent currently has.
        """
        self.logger.info('Updating state space')
        self.logger.debug(
            f'Opponent previous pos: {opp_previous_pos}, opponent next pos: {next_opponent_position}, house loc: {house_location}, agent money: {agent_money}')
        self.update_affordability_states(agent_money)
        self.state[OPPONENT_SPACE_INDEX, opp_previous_pos] = 0
        self.state[OPPONENT_SPACE_INDEX, next_opponent_position] = 1
        if self.board_positions[house_location].space_type == SpaceType.PROPERTY:
            self.state[HOUSES_INDEX, house_location] = 1

    def execute_action(self, house_location: int, agent_money: int, step: int):
        """
        Executes the action for a given house location, adjusting the agent's money and updating the game state.

        Args:
            house_location (int): The location of the house being bought.
            agent_money (int): The amount of money the agent has before the action.

        Returns:
            tuple: A tuple containing:
                - The rent charged to the agent (int).
                - The updated game state (np.ndarray).
                - A boolean indicating whether the game has ended (bool).
        """
        if step >= self.max_steps:
            self.logger.info(f'Max steps achieved! Ending game')
            return 0, self.state, True

        self.logger.debug(f'Step {step}')
        self.logger.info(f'Executing action with house location: {house_location}')
        game_end = False
        passed_go = False
        self.successes.append(0)

        position_data = self.board_positions[house_location]
        if not position_data.can_build_house():
            self.logger.debug(f'House cannot be bought at location: {house_location}. Cost is {self.default_cost}')
            house_cost = self.default_cost
        else:
            num_houses = position_data.num_houses
            house_cost = position_data.house_cost
            self.logger.debug(f'House can be bought at location: {house_location}. Cost is {house_cost}')
            if num_houses == 0:
                self.logger.debug('House not originally at location, adding house')
                position_data.num_houses += 1
            else:
                self.logger.debug('House already at location, not adding house')

        if agent_money - house_cost <= 0:
            # agent bankrupt
            self.logger.info('Agent is now bankrupt, ending game')
            game_end = True
            return -house_cost, self.opponent.curr_position, game_end

        rent = 0
        opponent_roll = self.opponent.get_action()
        opp_previous_pos = self.opponent.curr_position
        self.opponent.curr_position += opponent_roll
        if self.opponent.curr_position >= 40:
            # pass go
            passed_go = True
            self.opponent.curr_position = self.opponent.curr_position % 40
            self.opponent.money += 200
            self.logger.debug('Opponent passed GO, received 200 dollars')

        self.logger.debug(f'Opponent moving from {opp_previous_pos} to {self.opponent.curr_position}')

        opponent_position_data = self.board_positions[self.opponent.curr_position]
        if opponent_position_data.can_build_house():
            if opponent_position_data.num_houses > 0:
                rent = opponent_position_data.get_rent()
                self.logger.debug(f'Opponent landed on a property with a house, charging: {rent}')
                self.opponent.money -= rent
                self.successes[-1] = 1
                if self.opponent.money <= 0:
                    self.logger.info(f'Opponent is now bankrupt, ending game')
                    game_end = True

        # update state space here
        agent_money += rent - house_cost
        self.logger.debug(f'Agent payed ${house_cost} for house, and received ${rent} rent for a final total of ${agent_money}')
        self.update_state_space(opp_previous_pos, self.opponent.curr_position, house_location, agent_money)
        self.rewards.append(rent - house_cost)
        self.agent_monies.append(agent_money)
        self.opponent_monies.append(self.opponent.money)

        if passed_go:
            self.logger.debug('Passed go so resetting bought houses')
            self.state[HOUSES_INDEX] = np.zeros(40)

        return rent - house_cost, self.state, game_end
