import json
from typing import Dict

from board_space import BoardSpace
from space_type import SpaceType


class Board:
    def __init__(self, default_cost: int = -10):
        self.default_cost = -10
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
        position_data = self.board_positions[house_location]
        if position_data.space_type != SpaceType.PROPERTY:
            return self.default_cost

        num_houses = position_data.num_houses
        house_cost = position_data.house_cost
        if num_houses == 0:
            position_data.num_houses += 1

        # TODO add opponent's move and determine if lands on house
        return house_cost
