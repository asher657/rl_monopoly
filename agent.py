from board_space import BoardSpace
from space_type import SpaceType



class Agent:
    def __init__(self,
                 curr_space: BoardSpace = BoardSpace('go', 0, SpaceType.GO),
                 money: int = 1500):
        self.curr_space = curr_space
        self.money = money
        self.color_sets = set()
        # TODO: does owned properties go here or board

