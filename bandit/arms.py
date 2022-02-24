from typing import Union
import numpy as np


class ArmNotFoundException(Exception):
    def __init__(self, name):
        self.message = F'arm({name}) not found!'
        super().__init__(self.message)


class ArmAlreadyExistsException(Exception):
    def __init__(self, name):
        self.message = F'arm({name}) already exists!'
        super().__init__(self.message)


class Arm:

    def __init__(self, name: str):
        self.name = name
        self.selections = 0
        self.rewards = 0
        self.regrets = 0
        self.mean_reward = 0.0

    def update_mean_reward(self, reward):
        k = 1 / self.selections if self.selections > 0 else 0
        self.mean_reward = self.mean_reward + k * (reward - self.mean_reward)

    def reward(self, reward: Union[int, float] = None):
        self.rewards += reward
        self.update_mean_reward(reward)

    def select(self):
        self.selections += 1

    def __repr__(self):
        return F'Arm({self.name})'

    @classmethod
    def build(cls, name, weights=None):
        arm = cls(name=name)
        if weights:
            arm.__dict__.update(weights)
        return arm


class BernoulliArm(Arm):

    def __init__(self, name, p: float):
        super().__init__(name)
        self.p = p

    def draw(self):
        return int(np.random.choice([0, 1], p=[1-self.p, self.p]))
