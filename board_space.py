from typing import List
from enum import Enum, auto

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


PROPERTY_COLOR_COUNTS = {
    PropertyColor.BROWN: 2,
    PropertyColor.SKYBLUE: 3,
    PropertyColor.PURPLE: 3,
    PropertyColor.ORANGE: 3,
    PropertyColor.RED: 3,
    PropertyColor.YELLOW: 3,
    PropertyColor.GREEN: 3,
    PropertyColor.BLUE: 2
}


class PropertyType(Enum):
    LAND = auto()
    RAILROAD = auto()
    UTILITY = auto()


class BoardSpace:
    def __init__(self, name: str,
                 position: int,
                 space_type: str,
                 purchase_cost: int,
                 color: str,
                 property_type: str,
                 rent: List[int],
                 rent_hotel: int,
                 mortgage_value: int,
                 house_cost: int,
                 is_owned: bool = False,
                 is_mortgaged: bool = False,
                 num_houses: int = 0,
                 num_hotels: int = 0):
        self.name = name
        self.position = position
        self.space_type = SpaceType[space_type] if space_type is not None else None
        self.purchase_cost = purchase_cost
        self.color = PropertyColor[color] if color is not None else None
        self.property_type = PropertyType[property_type] if property_type is not None else None
        self.rent = rent
        self.rent_hotel = rent_hotel
        self.mortgage_value = mortgage_value
        self.mortgage_cost = mortgage_value * 1.1 if mortgage_value is not None else None
        self.house_cost = house_cost
        self.is_owned = is_owned
        self.is_mortgaged = is_mortgaged
        self.num_houses = num_houses
        self.num_hotels = num_hotels

    def get_rent(self):
        if self.property_type == PropertyType.LAND and self.num_houses == 1:
            return self.rent[self.num_houses]
        else:
            return 0
