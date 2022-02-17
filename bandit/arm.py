from typing import Union


class Arm:

    def __init__(self, name: str):
        self.name = name
        self.selections = 0
        self.rewards = 0

    @property
    def mean_reward(self):
        if self.selections == 0:
            return 0
        else:
            return self.rewards / self.selections

    def reward(self, amount: Union[int, float]):
        self.rewards += amount

    def select(self):
        self.selections += 1

    def __repr__(self):
        return f'Arm({self.name})'
