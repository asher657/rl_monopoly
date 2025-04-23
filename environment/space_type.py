from enum import Enum, auto


class SpaceType(Enum):
    FREE_SPACE = auto()
    PROPERTY = auto()
    CHANCE = auto()
    COMMUNITY_CHEST = auto()
    LUXURY_TAX = auto()
    INCOME_TAX = auto()
    GO_TO_JAIL = auto()
    JAIL = auto()
    GO = auto()

