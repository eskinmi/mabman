__all__ = [
    'EpsilonGreedy',
    'EpsilonDecay',
    'EpsilonFirst',
    'Hedge',
    'SoftmaxBoltzmann',
    'ThompsonSampling',
    'UCB1',
    'UCB2',
    'VDBE'
]

import random
import math
import numpy as np
from bandit.util import SuccessiveSelector
from bandit.agents.base import Agent
from bandit.arms import Arm
from typing import Optional, Union, List


class EpsilonGreedy(Agent):
    name = 'epsilon-greedy-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.1
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.epsilon = epsilon

    def selection_policy(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class EpsilonDecay(Agent):
    name = 'epsilon-decreasing-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.5,
                 gamma: float = 0.1,
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.epsilon = epsilon
        self.gamma = gamma

    def selection_policy(self):
        if random.random() > self.epsilon * (1 - self.gamma)**self.episode:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class EpsilonFirst(Agent):
    name = 'epsilon-first-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.1,
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.epsilon = epsilon
        self.start_exploration = self.episode * (1-self.epsilon) - 1

    def selection_policy(self):
        if self.episode >= self.start_exploration:
            chosen_arm = random.choice(self.active_arms)
        else:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        chosen_arm.select()

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class Hedge(Agent):
    name = 'hedge-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 temperature: Union[int, float] = 2
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.temperature = temperature

    def _threshold(self):
        return sum([math.exp(arm.rewards / self.temperature) for arm in self.active_arms])

    def selection_policy(self):
        th = self._threshold()
        z = random.random()
        chosen_arm = self.active_arms[-1]  # default
        p_sum = 0
        for arm in self.active_arms:
            p_sum += math.exp(arm.rewards / self.temperature) / th
            if p_sum > z:
                chosen_arm = arm
                break
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class SoftmaxBoltzmann(Agent):
    name = 'softmax-boltzmann-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 temperature: Union[int, float] = 2
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.temp = temperature

    def selection_policy(self):
        denominator = sum([math.exp(a.mean_reward / self.temp) for a in self.active_arms])
        probabilities = [math.exp(arm.mean_reward / self.temp) / denominator for arm in self.active_arms]
        chosen_arm = np.random.choice(self.active_arms, p=probabilities)
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class ThompsonSampling(Agent):
    name = 'thompson-sampling-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)

    def mk_draws(self):
        return [np.random.beta(arm.rewards + 1, arm.selections - arm.rewards + 1, size=1)
                for arm in self.active_arms
                ]

    def selection_policy(self):
        draws = self.mk_draws()
        chosen_arm = self.active_arms[draws.index(max(draws))]
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class UCB1(Agent):
    name = 'upper-confidence-bound-1-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 confidence: Union[int, float] = 2
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.confidence = confidence

    def calc_upper_bounds(self, arm):
        if arm.selections == 0:
            return 1e500
        else:
            return arm.mean_reward + (
                math.sqrt(self.confidence * math.log(self.episode + 1) / arm.selections)
            )

    def selection_policy(self):
        chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class UCB1Tuned(Agent):
    name = 'upper-confidence-bound-1-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 confidence: Union[int, float] = 2
                 ):
        self._arm_vars_hook(rewards_sq=0)
        self.confidence = confidence
        super().__init__(arms, episodes, reset_at_end, callbacks)

    def arm_variance_ub(self, arm):
        return (
            arm.rewards_sq / arm.selections - arm.mean_rewards ** 2 +
            math.sqrt(self.confidence * math.log(self.episode + 1) / arm.selections)
        )

    def calc_upper_bounds(self, arm):
        if arm.selections == 0:
            return 1e500
        else:
            return arm.mean_reward + (
                math.sqrt(math.log(self.episode + 1 / arm.selections) * min(0.25, self.arm_variance_ub(arm)))
            )

    def selection_policy(self):
        chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        arm = self.arm(name)
        arm.reward(reward)
        arm.rewards_sq += reward ** 2


class UCB2(Agent):
    name = 'upper-confidence-bound-2-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 alpha: float = 0.1
                 ):
        self.alpha = alpha
        self._arm_vars_hook(r=0)
        self._ss = SuccessiveSelector()
        super().__init__(arms, episodes, reset_at_end, callbacks)

    def calc_upper_bounds(self, arm):
        if arm.selections == 0:
            return 1e500
        else:
            tau = self.tau(arm.r)
            return (arm.mean_reward + (
                math.sqrt((1 + self.alpha) * math.log(math.e * self.episode / tau) / 2 * tau)
            )
                    )

    def calc_n_recursion(self, r):
        return self.tau(r+1) - self.tau(r)

    def tau(self, r):
        return math.ceil((1 + self.alpha) ** r)

    def selection_policy(self):
        if self._ss.in_recurrence:
            chosen_arm_name = self._ss.step()
            chosen_arm = self.arm(chosen_arm_name)
        else:
            chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
            self._ss.set_recurrence(chosen_arm.name, self.calc_n_recursion(chosen_arm.r))
            chosen_arm.r += 1
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)


class VDBE(Agent):
    name = 'epsilon-greedy-vdbe-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 sigma: float = 0.5,
                 init_epsilon: float = 0.3
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.sigma = sigma
        self.init_epsilon = init_epsilon
        self._prev_epsilon = self.init_epsilon
        self._previous_mean_reward = self.agent_mean_reward

    @property
    def delta(self):
        if self.episode != 0:
            return 1 / self.episode
        else:
            return 1

    @property
    def action_value(self):
        prior = 1 - math.exp(-1 * abs(self.agent_mean_reward - self._previous_mean_reward) / self.sigma)
        return (1 - prior) / (1 + prior)

    @property
    def epsilon(self):
        return self.delta * self.action_value + (1 - self.delta) * self._prev_epsilon

    def selection_policy(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self._prev_epsilon = self.epsilon
        self._previous_mean_reward = self.agent_mean_reward
        self.arm(name).reward(reward)
