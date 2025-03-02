from typing import List
from enum import Enum, auto

from board_space import BoardSpace
from space_type import SpaceType


class PropertyColor(Enum):
    BROWN = auto()
    SKYBLUE = auto()
    PURPLE = auto()
    ORANGE = auto()
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    BLUE = auto()


class PropertyType(Enum):
    LAND = auto()
    RAILROAD = auto()
    UTILITY = auto()


class Property(BoardSpace):
    def __init__(self, name: str,
                 position: int,
                 purchase_cost: int,
                 color: PropertyColor,
                 property_type: PropertyType,
                 rent: List[int],
                 rent_hotel: int,
                 mortgage_value: int,
                 house_cost: int,
                 is_owned: bool = False,
                 is_mortgaged: bool = False,
                 num_houses: int = 0,
                 num_hotels: int = 0):
        super().__init__(name, position, SpaceType.PROPERTY)
        self.purchase_cost = purchase_cost
        self.color = color
        self.property_type = property_type
        self.rent = rent
        self.rent_hotel = rent_hotel
        self.mortgage_value = mortgage_value
        self.mortgage_cost = self.mortgage_value * 1.1
        self.house_cost = house_cost
        self.is_owned = is_owned
        self.is_mortgaged = is_mortgaged
        self.num_houses = num_houses
        self.num_hotels = num_hotels

    def get_rent(self, property_type: PropertyType, dice_roll: int):
        if property_type == PropertyType.LAND:
            if self.is_mortgaged or not self.is_owned:
                return 0
            if self.num_hotels > 0:
                return self.rent_hotel
            else:
                return self.rent[self.num_houses]
        elif property_type == PropertyType.RAILROAD:
            # TODO: get number of railroads owned and return value
            num_railroads = None
            return self.rent[num_railroads - 1]
        else:
            # TODO: get number of utilities owned and return value
            num_utilities = None
            return dice_roll * self.rent[num_utilities]
