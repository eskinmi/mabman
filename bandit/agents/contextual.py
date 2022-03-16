__all__ = [
    'LinUCB'
]

import numpy as np
from bandit.agents.base import Agent
from bandit.arms import Arm
from bandit.states import StatesSecretary
from typing import Optional, Union, List


class LinUCB(Agent):
    name = 'linear-upper-confidence-bound-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 d: int = 100,
                 alpha: float = 1.5
                 ):
        self._arm_vars_hook(A=np.identity(d), b=np.zeros([d, 1]))
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.statesman = StatesSecretary()
        self.alpha = alpha
        self.d = d

    def arm_upper_bound(self, arm, context):
        a_inv = np.linalg.inv(arm.A)
        theta = np.dot(a_inv, arm.b)
        return np.dot(theta.T, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(a_inv, context)))

    def selection_policy(self, context):
        with self.statesman.instate(context) as c:
            chosen_arm = max(self.active_arms, key=lambda x: self.arm_upper_bound(x, c))
            chosen_arm.select()
            return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        state = self.statesman.last_state
        arm = self.arm(name)
        arm.reward(reward)
        arm.A += np.dot(state, state.T)
        arm.b += reward*state
