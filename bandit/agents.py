__all__ = [
    'EpsilonGreedy',
    'EpsilonDecay',
    'EpsilonFirst',
    'Hedge',
    'SoftmaxBoltzmann',
    'ThompsonSampling',
    'UCB1',
    'UCB2',
    'VDBE',
    'EXP3',
    'FPL'
]

import random
import math
import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod
from bandit import process
from bandit.arms import Arm, ArmNotFoundException, ArmAlreadyExistsException
from bandit.callbacks import WrongBanditCheckPointError, CheckPointState


class MissingRewardException(Exception):
    def __init__(self, episode: int):
        self.message = F'round {episode} is not rewarded.'
        super().__init__(self.message)


class SuccessiveSelector:

    def __init__(self):
        self.arm_name = None
        self.n = 0

    @property
    def in_recurrence(self):
        if self.arm_name is not None and self.n > 0:
            return True
        else:
            return False

    def set_recurrence(self, name, n):
        self.arm_name = name
        self.n = n

    def step(self):
        if self.in_recurrence:
            self.n -= 1
            return self.arm_name
        else:
            self.n = 0
            self.arm_name = None


class Agent(process.Process, ABC):

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.arms = []
        self.init_arm_vars = dict()

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def choose_arm(self):
        pass

    @abstractmethod
    def reward_arm(self, name: str, reward: Union[int, float]):
        pass

    @property
    def active_arms(self):
        return list(filter(lambda x: x.active, self.arms))

    @property
    def agent_mean_reward(self):
        if self.episode > 0:
            return self.total_rewards / self.episode
        else:
            return 0

    @property
    def total_rewards(self):
        return sum([arm.rewards for arm in self.arms])

    @property
    def total_selections(self):
        return sum([arm.selections for arm in self.arms])

    @property
    def arm_names(self):
        return [arm.name for arm in self.arms]

    @property
    def is_choice_made(self):
        return self.total_selections == self.episode + 1

    @property
    def episode_closed(self):
        return self.total_selections == self.episode

    def set_init_arm_attrs(self, **kwargs):
        self.init_arm_vars = kwargs

    def _update_attrs(self,  params: dict):
        self.__dict__.update(params)

    def choose(self):
        if not self.stop and self.episode_closed:
            return self.choose_arm()
        else:
            raise MissingRewardException(self.episode)

    def reward(self, name: str, reward: Union[int, float] = 1):
        if self.is_choice_made:
            self.reward_arm(name, reward)
            self.add_episode_logs(name, reward, self.arm_names)
            self.proceed()
        else:
            raise MissingRewardException(self.episode)

    def arm(self, name: str):
        if name in self.arm_names:
            return self.arms[self.arm_names.index(name)]
        else:
            raise ArmNotFoundException(name)

    def add_arm(self, arm: Arm):
        if arm.name not in self.arm_names:
            [setattr(arm, k, v) for k, v in self.init_arm_vars.items()]
            self.arms.append(arm)
        else:
            raise ArmAlreadyExistsException(arm.name)

    def deactivate_arm(self, name: str):
        self.arm(name).active = False

    def overlay_weights(self, path):
        ckp = CheckPointState(path)
        arms_weights, exp_params, agent_params = ckp.load_component_weights()
        if self.name == agent_params['name']:
            self.arms = [
                Arm.build(arm_weights['name'], arm_weights['weights'])
                for arm_weights in arms_weights
            ]
            self.experiment = process.Experiment.build(exp_params)
            self._update_attrs(agent_params['params'])
        else:
            raise WrongBanditCheckPointError(agent_params['name'])


class EpsilonGreedy(Agent):
    name = 'epsilon-greedy-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.epsilon = epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class EpsilonDecay(Agent):
    name = 'epsilon-decreasing-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.5,
                 gamma: float = 0.1,
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_arm(self):
        if random.random() > self.epsilon * (1 - self.gamma)**self.episode:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class EpsilonFirst(Agent):
    name = 'epsilon-first-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.1,
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.epsilon = epsilon
        self.start_exploration = self.episode * (1-self.epsilon) - 1

    def choose_arm(self):
        if self.episode >= self.start_exploration:
            chosen_arm = random.choice(self.active_arms)
        else:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        chosen_arm.select()

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class Hedge(Agent):
    name = 'hedge-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 temperature: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.temperature = temperature

    def _threshold(self):
        return sum([math.exp(arm.rewards / self.temperature) for arm in self.active_arms])

    def choose_arm(self):
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

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class SoftmaxBoltzmann(Agent):
    name = 'softmax-boltzmann-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 temperature: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.temp = temperature

    def choose_arm(self):
        denominator = sum([math.exp(a.mean_reward / self.temp) for a in self.active_arms])
        probabilities = [math.exp(arm.mean_reward / self.temp) / denominator for arm in self.active_arms]
        chosen_arm = np.random.choice(self.active_arms, p=probabilities)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class ThompsonSampling(Agent):
    name = 'thompson-sampling-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 ):
        super().__init__(episodes, reset_at_end, callbacks)

    def mk_draws(self):
        return [np.random.beta(arm.rewards + 1, arm.selections - arm.rewards + 1, size=1)
                for arm in self.active_arms
                ]

    def choose_arm(self):
        draws = self.mk_draws()
        chosen_arm = self.active_arms[draws.index(max(draws))]
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class UCB1(Agent):
    name = 'upper-confidence-bound-1-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 confidence: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.confidence = confidence

    def calc_upper_bounds(self, arm):
        if arm.selections == 0:
            return 1e500
        else:
            return arm.mean_reward + (
                    self.confidence * math.sqrt(math.log(self.episode + 1) / arm.selections)
            )

    def choose_arm(self):
        chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class UCB2(Agent):
    name = 'upper-confidence-bound-2-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 alpha: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.alpha = alpha
        self.set_init_arm_attrs(r=0)
        self._ss = SuccessiveSelector()

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

    def choose_arm(self):
        if self._ss.in_recurrence:
            chosen_arm_name = self._ss.step()
            chosen_arm = self.arm(chosen_arm_name)
        else:
            chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
            self._ss.set_recurrence(chosen_arm.name, self.calc_n_recursion(chosen_arm.r))
            chosen_arm.r += 1
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class VDBE(Agent):
    name = 'epsilon-greedy-vdbe-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 sigma: float = 0.5,
                 init_epsilon: float = 0.3
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
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

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self._prev_epsilon = self.epsilon
        self._previous_mean_reward = self.agent_mean_reward
        self.arm(name).reward(reward)


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
        self.set_init_arm_attrs(weight=1)

    def _update_arm_weight(self, arm, reward):
        estimate = reward / self._arm_proba(arm)
        arm.weight *= math.exp(estimate * self.gamma / len(self.active_arms))

    def _arm_proba(self, arm):
        return (1.0 - self.gamma) * (arm.weight / self._w_sum()) + (self.gamma / len(self.active_arms))

    def _w_sum(self):
        return sum([arm.weight for arm in self.active_arms])

    def choose_arm(self):
        w_dist = [self._arm_proba(arm) for arm in self.active_arms]
        chosen_arm = np.random.choice(self.active_arms, p=w_dist)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        arm = self.arm(name)
        self._update_arm_weight(arm, reward)
        arm.reward(reward)


class FPL(Agent):
    name = 'follow-perturbed-leader-bandit'

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 noise_param: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end, callbacks)
        self.noise_param = noise_param

    def _noise(self):
        return float(np.random.exponential(self.noise_param))

    def choose_arm(self):
        chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward + self._noise())
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)
