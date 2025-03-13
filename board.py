import json
from typing import Dict

from agent import Agent
from board_space import BoardSpace
from opponent import Opponent
from space_type import SpaceType


class Board:
    def __init__(self, agent: Agent, default_cost: int = -10, debug: bool = False):
        self.agent = agent
        self.default_cost = default_cost
        self.opponent = Opponent(debug)
        self.board_positions = self._load_positions()
        self.bought_houses = {i: 0 for i in range(len(self.board_positions))}

    def _load_positions(self) -> Dict[int, BoardSpace]:
        with open('positions.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        return {int(key): BoardSpace(**value) for key, value in json_data.items()}

    def get_rent_cost(self, position: int):
        position_data = self.board_positions[position]
        return position_data.get_rent()

    def execute_action(self, house_location: int):
        reward = 0
        house_cost = 0

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
            return reward - house_cost, self.opponent.curr_position, True

        rent = 0
        opponent_roll = self.opponent.get_action()
        self.opponent.curr_position += opponent_roll
        if opponent_roll >= 40:
            # pass go
            self.opponent.curr_position = opponent_roll % 40
            self.opponent.money += 200

        opponent_position_data = self.board_positions[self.opponent.curr_position]
        if opponent_position_data.space_type == SpaceType.PROPERTY:
            if opponent_position_data.num_houses > 0:
                rent = opponent_position_data.get_rent()

        reward = reward + rent - house_cost

        return reward, self.opponent.curr_position, False
