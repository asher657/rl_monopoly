import json
import os.path
from typing import Dict
import numpy as np

from agent import Agent
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
        self.state = np.zeros(shape=(5, 40), dtype=np.int32)

        self.bought_positions = []
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
        self.logger.debug('Updating affordability states')
        for idx, space in self.board_positions.items():
            if space.can_build_house():
                if agent_money > space.rent[INITIAL_NUM_HOUSES]:
                    self.state[PLAYER_MONEY_INDEX, idx] = 1
                else:
                    self.state[PLAYER_MONEY_INDEX, idx] = 0
                if self.opponent.money > space.rent[INITIAL_NUM_HOUSES]:
                    self.state[OPPONENT_MONEY_INDEX, idx] = 1
                else:
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
        self.logger.debug('Updating state space')
        self.update_affordability_states(agent_money)
        self.state[OPPONENT_SPACE_INDEX, opp_previous_pos] = 0
        self.state[OPPONENT_SPACE_INDEX, next_opponent_position] = 1

    def execute_action(self, house_location: int, agent: Agent, step: int):
        """
        Executes the action for a given house location, adjusting the agent's money and updating the game state.

        Args:
            house_location (int): The location of the house being bought.
            agent (Agent): The agent used in this run.
            step (int): The current step of the action

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
        self.successes.append(0)
        self.bought_positions.append(house_location)
        not_house_eligible = False

        position_data = self.board_positions[house_location]
        if not position_data.can_build_house():
            self.logger.debug(f'House cannot be bought at location: {house_location}. Default cost is ${-self.default_cost}')
            house_cost = self.default_cost
            not_house_eligible = True
        else:
            house_cost = position_data.house_cost
            if self.state[HOUSES_INDEX, house_location] < 4:
                self.state[HOUSES_INDEX, house_location] += 1
                self.logger.debug(f'Buying house at location: {house_location}. Cost is ${house_cost}. {self.state[HOUSES_INDEX, house_location]} houses at this location')
            else:
                self.logger.debug(f'Max houses reached at location, not adding house. Default cost is {self.default_cost}')
                house_cost = self.default_cost
                not_house_eligible = True

        rent = 0
        opponent_roll = self.opponent.get_action()
        opp_previous_pos = self.opponent.curr_position
        self.opponent.curr_position += opponent_roll
        if self.opponent.curr_position >= 40:
            # pass go
            self.opponent.curr_position = self.opponent.curr_position % 40
            self.opponent.money += 200
            self.logger.debug('Opponent passed GO, received 200 dollars')

        self.logger.debug(f'Opponent moving from {opp_previous_pos} to {self.opponent.curr_position}')

        opponent_position_data = self.board_positions[self.opponent.curr_position]
        if opponent_position_data.space_type == SpaceType.LUXURY_TAX:
            self.opponent.money -= LUXURY_TAX_COST
            self.logger.info(
                f'Opponent must pay luxury tax of {LUXURY_TAX_COST}. Opponent money is now: ${self.opponent.money}')
        elif opponent_position_data.space_type == SpaceType.INCOME_TAX:
            self.opponent.money -= INCOME_TAX_COST
            self.logger.info(
                f'Opponent must pay income tax of {INCOME_TAX_COST}. Opponent money is now: ${self.opponent.money}')
        elif opponent_position_data.space_type == SpaceType.GO_TO_JAIL:
            self.opponent.money -= GO_TO_JAIL_COST
            self.opponent.curr_position = JAIL_POSITION
            self.logger.info(
                f'Opponent went to jail and must pay {GO_TO_JAIL_COST} to get out. Opponent money is now: ${self.opponent.money}')
        else:
            if self.state[HOUSES_INDEX, self.opponent.curr_position] == 0 or not opponent_position_data.can_build_house():
                self.logger.debug('Opponent did not land on a position with a house. No rent is paid')
            elif opponent_position_data.can_build_house():
                if self.state[HOUSES_INDEX, self.opponent.curr_position] > 0:
                    rent = opponent_position_data.get_rent(self.state[HOUSES_INDEX, self.opponent.curr_position])
                    self.logger.debug(f'Opponent landed on a property with a house, charging: ${rent}')
                    self.opponent.money -= rent
                    self.successes[-1] = 1
                    if self.opponent.money <= 0:
                        self.logger.info(f'Opponent is now bankrupt, ending game')
                        game_end = True

        # update state space here
        if not_house_eligible:
            agent.money += rent
        else:
            agent.money += rent - house_cost
        self.logger.info(f'Agent paid ${house_cost} for house, and received ${rent} rent. Agent money is now: ${agent.money}')
        self.update_state_space(opp_previous_pos, self.opponent.curr_position, house_location, agent.money)
        self.rewards.append(rent - house_cost)
        self.agent_monies.append(agent.money)
        self.opponent_monies.append(self.opponent.money)

        if np.all(self.state[PLAYER_MONEY_INDEX] == 0):
            self.logger.info('Agent is now bankrupt or can no longer afford any houses, ending game')
            game_end = True

        if np.all(self.state[HOUSES_INDEX][self.state[RENT_INDEX] > 0] == 4):
            self.logger.info('==== All houses bought! ====')

        self.logger.info(
            f'Opponent previous pos: {opp_previous_pos}, opponent next pos: {self.opponent.curr_position}, house loc: {house_location}, agent money: ${agent.money}, opponent money: ${self.opponent.money}')

        if house_location == self.opponent.curr_position:
            reward = 2 * rent - house_cost
        elif rent > 0:
            reward = rent - house_cost
        else:
            reward = -house_cost
        self.logger.info(f'Reward: {reward}')
        return reward, self.state, game_end
