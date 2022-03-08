__all__ = [
    'LinUCB'
]

import numpy as np
from bandit.agents.base import Agent
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
        self.d = d
        self._set_init_arm_attrs(A=np.identity(d), b=np.zeros([d, 1]))

    def arm_upper_bound(self, arm, context):
        a_inv = np.linalg.inv(arm.A)
        theta = np.dot(a_inv, arm.b)
        return np.dot(theta.T, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(a_inv, context)))

    def choose_arm(self, context=None):
        with self.env.instate(context) as c:
            chosen_arm = max(self.active_arms, key=lambda x: self.arm_upper_bound(x, c))
            chosen_arm.select()
            return chosen_arm.name

    def reward_arm(self, name: str, reward: Union[int, float]):
        state = self.env.last_state
        arm = self.arm(name)
        arm.reward(reward)
        arm.A += np.dot(state, state.T)
        arm.b += reward*state
