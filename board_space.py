from typing import List
from enum import Enum, auto

from space_type import SpaceType


class PropertyColor(Enum):
    """Enumeration for different property colors in Monopoly."""
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
    """Enumeration for different types of properties in Monopoly."""
    LAND = auto()
    RAILROAD = auto()
    UTILITY = auto()


class BoardSpace:
    """
    Represents a space on the Monopoly board.

    Attributes:
        name (str): The name of the board space.
        position (int): The index position of the space on the board.
        space_type (SpaceType): The type of space (e.g., PROPERTY, CHANCE, GO, etc.).
        purchase_cost (int): The cost to purchase the property, if applicable.
        color (PropertyColor or None): The color group of the property, if applicable.
        property_type (PropertyType or None): The type of property (LAND, RAILROAD, UTILITY).
        rent (List[int]): The rent values based on the number of houses built.
        rent_hotel (int): The rent when a hotel is placed.
        mortgage_value (int): The amount received when mortgaging the property.
        mortgage_cost (float or None): The cost to unmortgage the property (110% of mortgage_value).
        house_cost (int): The cost to build a house on the property.
        is_owned (bool): Whether the property is currently owned.
        is_mortgaged (bool): Whether the property is mortgaged.
        num_houses (int): The number of houses built on the property.
        num_hotels (int): The number of hotels built on the property.
    """
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

    def get_rent(self) -> int:
        """
        Determines the rent amount based on the number of houses built.

        Returns:
            int: The rent amount. Returns 0 if the property is not a LAND property.
        """
        if self.property_type == PropertyType.LAND and self.num_houses == 1:
            return self.rent[self.num_houses]
        else:
            return 0

    def can_build_house(self) -> bool:
        """
        Checks if a house can be built on this property.

        Returns:
            bool: True if the property is a buildable LAND property, False otherwise.
        """
        return self.space_type == SpaceType.PROPERTY and self.property_type == PropertyType.LAND

