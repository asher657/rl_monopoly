import random

from board_space import BoardSpace
from property import Property, PropertyType, PropertyColor, PROPERTY_COLOR_COUNTS
from space_type import SpaceType

initial_color_group_counts = {PropertyColor.BROWN: 0,
                              PropertyColor.SKYBLUE: 0,
                              PropertyColor.PURPLE: 0,
                              PropertyColor.ORANGE: 0,
                              PropertyColor.RED: 0,
                              PropertyColor.YELLOW: 0,
                              PropertyColor.GREEN: 0,
                              PropertyColor.BLUE: 0}

class Player:
    def __init__(self,
                 curr_space: BoardSpace = BoardSpace('go', 0, SpaceType.GO),
                 money: int = 1500):
        self.curr_space = curr_space
        self.money = money
        self.owned_color_groups = set()
        self.color_group_counts = initial_color_group_counts
        self.owned_properties = set()

    def make_move(self):
        move = random.randint(1, 12)
        self.curr_space += move # TODO: wrap around if they get to GO
        # TODO: handle space logic

    def buy_property(self, bought_property: Property):
        self.owned_properties.add(bought_property)
        if bought_property.property_type == PropertyType.LAND:
            self.set_color_groups(bought_property)

    def set_color_groups(self, bought_property):
        property_color = bought_property.color
        self.color_group_counts[property_color] += 1
        if self.color_group_counts[property_color] == PROPERTY_COLOR_COUNTS[property_color]:
            self.owned_color_groups.add(property_color)



