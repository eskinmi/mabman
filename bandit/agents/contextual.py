__all__ = [
    'LinUCB'
]

import numpy as np
from bandit.agents.base import Agent, MissingRewardException
from typing import Optional, Union


class LinUCB(Agent):
    name = 'linear-upper-confidence-bound-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 d: int = 100,
                 alpha: float = 1.5
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.alpha = alpha
        self._set_init_arm_attrs(A=np.identity(d), b=np.zeros([d, 1]))
        self.x = None

    def _set_context(self, x: np.array):
        if self.episode_closed:
            self.x = x.reshape([-1, 1])
        else:
            raise MissingRewardException(self.episode)

    def calc_upper_bounds(self, arm):
        a_inv = np.linalg.inv(arm.A)
        theta = np.dot(a_inv, arm.b)
        ucb = np.dot(theta.T, self.x) + self.alpha * np.sqrt(np.dot(self.x.T, np.dot(a_inv, self.x)))
        return ucb

    def choose_arm(self, context):
        self._set_context(context)
        chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward: Union[int, float]):
        arm = self.arm(name)
        arm.reward(reward)
        arm.A += np.dot(self.x, self.x.T)
        arm.b += reward*self.x
