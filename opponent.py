import numpy as np


class Opponent:
    def __init__(self, money: int = 1500, debug=False):
        self.money = money
        self.debug = debug

    def get_action(self):
        roll_1 = np.random.randint(1, 7)
        roll_2 = np.random.randint(1, 7)
        if self.debug:
            print(f'Opponent rolled {roll_1} and {roll_2} with total {roll_1 + roll_2}')
        return roll_1 + roll_2
