from bandit.agents.base import *
import math
import numpy as np


class SoftmaxBoltzmann(Agent):
    name = 'softmax-boltzmann-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 temperature
                 ):
        super().__init__(episodes, reset_at_end)
        self.temp = temperature

    def choose_arm(self):
        denominator = sum([math.exp(a.mean_reward / self.temp) for a in self.arms])
        probabilities = [math.exp(arm.mean_reward / self.temp) / denominator for arm in self.arms]
        chosen_arm = np.random.choice(self.arms, p=probabilities)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)
