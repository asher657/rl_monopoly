from board_space import BoardSpace
from space_type import SpaceType


class Property(BoardSpace):
    def __init__(self, name: str,
                 position: int,
                 purchase_cost: int,
                 color: str,
                 rent: int,
                 rent_1_house: int,
                 rent_2_house: int,
                 rent_3_house: int,
                 rent_4_house: int,
                 rent_hotel: int,
                 mortgage_value: int,
                 house_cost: int,
                 hotel_cost: int,
                 is_owned: bool = False,
                 is_mortgaged: bool = False,
                 num_houses: int = 0,
                 num_hotels: int = 0):
        super().__init__(name, position, SpaceType.PROPERTY)
        self.purchase_cost = purchase_cost
        self.color = color
        self.rent = rent
        self.rent_1_house = rent_1_house
        self.rent_2_house = rent_2_house
        self.rent_3_house = rent_3_house
        self.rent_4_house = rent_4_house
        self.rent_hotel = rent_hotel
        self.mortgage_value = mortgage_value
        self.house_cost = house_cost
        self.hotel_cost = hotel_cost
        self.is_owned = is_owned
        self.is_mortgaged = is_mortgaged
        self.num_houses = num_houses
        self.num_hotels = num_hotels

    def get_rent(self):
        if self.is_mortgaged or not self.is_owned:
            return 0
        if self.num_houses == 0 and self.num_hotels == 0:
            return self.rent
        elif self.num_hotels >= 1:
            return self.rent_hotel
        elif self.num_houses == 1:
            return self.rent_1_house
        elif self.num_houses == 2:
            return self.rent_2_house
        elif self.num_houses == 3:
            return self.rent_3_house
        elif self.num_houses == 4:
            return self.rent_4_house

