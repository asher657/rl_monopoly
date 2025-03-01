from enum import Enum, auto


class SpaceType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    FREE_SPACE = auto()
    PROPERTY = auto()
    CHANCE = auto()
    COMMUNITY_CHEST = auto()
    LUXURY_TAX = auto()
    INCOME_TAX = auto()
    GO_TO_JAIL = auto()
    JAIL = auto()
    GO = auto()

