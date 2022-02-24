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

    def __init__(self, name: str, p: float = None):
        self.name = name
        self.p = p
        self.selections = 0
        self.rewards = 0
        self.regrets = 0
        self.mean_reward = 0.0
        self.active = True

    def update_mean_reward(self, reward):
        k = 1 / self.selections if self.selections > 0 else 0
        self.mean_reward = self.mean_reward + k * (reward - self.mean_reward)

    def reward(self, reward: Union[int, float] = None):
        self.rewards += reward
        self.update_mean_reward(reward)

    def select(self):
        self.selections += 1

    def draw(self):
        if self.p:
            return int(np.random.choice([0, 1], p=[1-self.p, self.p]))
        else:
            raise ValueError('please input the probability argument `p` to the arm.')

    def __repr__(self):
        return F'Arm({self.name})'

    @classmethod
    def build(cls, name, weights=None):
        arm = cls(name=name)
        if weights:
            arm.__dict__.update(weights)
        return arm
