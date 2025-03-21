import json
from typing import Dict
import numpy as np

from agent import Agent
from board_space import BoardSpace
from opponent import Opponent
from space_type import SpaceType

HOUSES_INDEX = 0
OPPONENT_SPACE_INDEX = 1
RENT_INDEX = 2
PLAYER_MONEY_INDEX = 3
OPPONENT_MONEY_INDEX = 4

INITIAL_NUM_HOUSES = 1


class Board:
    def __init__(self, agent: Agent, default_cost: int = -10, debug: bool = False):
        self.agent = agent
        self.default_cost = default_cost
        self.debug = debug
        self.opponent = Opponent(self.debug)
        self.board_positions = self._load_positions()
        self.bought_houses = {i: 0 for i in range(len(self.board_positions))}
        self.state = np.zeros(shape=(5, 40))

    def _load_positions(self) -> Dict[int, BoardSpace]:
        with open('positions.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        return {int(key): BoardSpace(**value) for key, value in json_data.items()}

    def reset(self, agent: Agent):
        self.agent = agent
        self.opponent = Opponent(self.debug)
        for idx, space in self.board_positions.items():
            if space.space_type == SpaceType.PROPERTY:
                self.state[RENT_INDEX, idx] = space.rent[INITIAL_NUM_HOUSES]
        self.state[OPPONENT_SPACE_INDEX, self.opponent.curr_position] = 1
        self.update_affordability_states()

    def update_affordability_states(self) -> None:
        for idx, space in self.board_positions.items():
            if space.space_type == SpaceType.PROPERTY:
                if self.agent.money > space.rent[INITIAL_NUM_HOUSES]:
                    self.state[PLAYER_MONEY_INDEX, idx] = 1
                else:
                    self.state[PLAYER_MONEY_INDEX, idx] = 0
                if self.opponent.money > space.rent[INITIAL_NUM_HOUSES]:
                    self.state[OPPONENT_MONEY_INDEX, idx] = 1
                else:
                    self.state[OPPONENT_MONEY_INDEX, idx] = 0
        return None

    def update_state_space(self, agent: Agent, opponent: Opponent, opp_previous_pos, next_opponent_position: int, house_location: int) -> None:
        self.update_affordability_states(agent, opponent)
        self.state[OPPONENT_SPACE_INDEX, opp_previous_pos] = 0
        self.state[OPPONENT_SPACE_INDEX, next_opponent_position] = 1
        self.state[HOUSES_INDEX, house_location] = 1
        return None

    def execute_action(self, house_location: int):
        game_end = False

        position_data = self.board_positions[house_location]
        if position_data.space_type != SpaceType.PROPERTY:
            house_cost = self.default_cost
        else:
            num_houses = position_data.num_houses
            house_cost = position_data.house_cost
            if num_houses == 0:
                position_data.num_houses += 1

        if self.agent.money - house_cost <= 0:
            # agent bankrupt
            game_end = True
            return -house_cost, self.opponent.curr_position, game_end

        rent = 0
        opponent_roll = self.opponent.get_action()
        opp_previous_pos = self.opponent.curr_position
        self.opponent.curr_position += opponent_roll
        if self.opponent.curr_position >= 40:
            # pass go
            self.opponent.curr_position = self.opponent.curr_position % 40
            self.opponent.money += 200

        opponent_position_data = self.board_positions[self.opponent.curr_position]
        if opponent_position_data.space_type == SpaceType.PROPERTY:
            if opponent_position_data.num_houses > 0:
                rent = opponent_position_data.get_rent()
                self.opponent.money -= rent
                if self.opponent.money <= 0:
                    game_end = True

        # update state space here
        self.update_state_space(self.agent, self.opponent, opp_previous_pos, self.opponent.curr_position, house_location)
        return rent - house_cost, self.opponent.curr_position, game_end
