from typing import Union


class Arm:

    def __init__(self, name: str):
        self.name = name
        self.selections = 0
        self.rewards = 0
        self.mean_reward = 0.0

    def update_mean_reward(self, reward):
        k = 1 / self.selections if self.selections > 0 else 0
        self.mean_reward = self.mean_reward + k * (reward - self.mean_reward)

    def reward(self, reward: Union[int, float]):
        self.rewards += reward
        self.update_mean_reward(reward)

    def select(self):
        self.selections += 1

    def __repr__(self):
        return f'Arm({self.name})'


class ArmNotFoundException(Exception):
    def __init__(self, name):
        self.message = F'arm({name}) not found!'
        super().__init__(self.message)