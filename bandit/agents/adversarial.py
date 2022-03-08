__all__ = [
    'EXP3',
    'FPL'
]

from bandit.agents.base import Agent
import numpy as np
import math
from typing import Optional, Union


class EXP3(Agent):
    name = 'exponential-weight-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 gamma: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.gamma = gamma
        self._set_init_arm_attrs(weight=1)

    def _update_arm_weight(self, arm, reward):
        estimate = reward / self._arm_proba(arm)
        arm.weight *= math.exp(estimate * self.gamma / len(self.active_arms))

    def _arm_proba(self, arm):
        return (1.0 - self.gamma) * (arm.weight / self._w_sum()) + (self.gamma / len(self.active_arms))

    def _w_sum(self):
        return sum([arm.weight for arm in self.active_arms])

    def choose_arm(self, context=None):
        w_dist = [self._arm_proba(arm) for arm in self.active_arms]
        chosen_arm = np.random.choice(self.active_arms, p=w_dist)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward: Union[int, float]):
        arm = self.arm(name)
        self._update_arm_weight(arm, reward)
        arm.reward(reward)


class FPL(Agent):
    name = 'follow-perturbed-leader-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 noise_param: float = 5
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.noise_param = noise_param

    def _noise(self):
        return float(np.random.exponential(self.noise_param))

    def choose_arm(self, context=None):
        chosen_arm = max(self.active_arms, key=lambda x: x.rewards + self._noise())
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)
