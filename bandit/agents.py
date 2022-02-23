from typing import Union
from abc import ABC, abstractmethod
from bandit import process
from bandit.arms import Arm, ArmNotFoundException, ArmAlreadyExistsException
import random
import math
import numpy as np


class MissingRewardException(Exception):
    def __init__(self, episode: int):
        self.message = F'round {episode} is not rewarded.'
        super().__init__(self.message)


class Agent(process.Process, ABC):

    def __init__(self,
                 episodes: int,
                 reset_at_end: bool
                 ):
        super().__init__(episodes, reset_at_end)
        self.arms = []

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

    def choose(self):
        if not self.stop and self.episode_closed:
            return self.choose_arm()

    def reward(self, name: str, reward: Union[int, float] = 1):
        if self.is_choice_made:
            self.reward_arm(name, reward)
            self.log_episode(name, reward, self.arm_names)
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
            self.arms.append(arm)
        else:
            raise ArmAlreadyExistsException(arm.name)

    def remove_arm(self, name: str):
        self.arms = [arm for arm in self.arms if arm.name != name]


class EpsilonGreedy(Agent):
    name = 'epsilon-greedy-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 epsilon: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class EpsilonDecay(Agent):
    name = 'epsilon-decreasing-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 epsilon: float = 0.5,
                 gamma: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_arm(self):
        if random.random() > self.epsilon * (1-self.gamma)**self.episode:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class EpsilonFirst(Agent):
    name = 'epsilon-first-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 epsilon: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon
        self.start_exploration = self.episode * (1-self.epsilon) - 1

    def choose_arm(self):
        if self.episode >= self.start_exploration:
            chosen_arm = random.choice(self.arms)
        else:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        chosen_arm.select()

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class Hedge(Agent):
    name = 'hedge-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 temperature: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end)
        self.temperature = temperature

    def _threshold(self):
        return sum([math.exp(arm.rewards / self.temperature) for arm in self.arms])

    def choose_arm(self):
        th = self._threshold()
        z = random.random()
        chosen_arm = self.arms[-1]  # default
        p_sum = 0
        for arm in self.arms:
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


class ThompsonSampling(Agent):
    name = 'thompson-sampling-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end
                 ):
        super().__init__(episodes, reset_at_end)

    def mk_draws(self):
        return [np.random.beta(arm.rewards + 1, arm.selections - arm.rewards + 1, size=1)
                for arm in self.arms
                ]

    def choose_arm(self):
        draws = self.mk_draws()
        chosen_arm = self.arms[draws.index(max(draws))]
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class UCB1(Agent):
    name = 'upper-confidence-bound-1-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 confidence: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end)
        self.confidence = confidence

    def calc_upper_bounds(self, arm):
        if arm.selections == 0:
            return 1e500
        else:
            return arm.mean_reward + (
                    self.confidence * math.sqrt(math.log(self.episode + 1) / arm.selections)
            )

    def choose_arm(self):
        chosen_arm = max(self.arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class VDBE(Agent):
    name = 'epsilon-greedy-vdbe-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 sigma,
                 init_epsilon=0.3
                 ):
        super().__init__(episodes, reset_at_end)
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
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self._prev_epsilon = self.epsilon
        self._previous_mean_reward = self.agent_mean_reward
        self.arm(name).reward(reward)


class EXP3(Agent):
    name = 'exponential-weight-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 gamma
                 ):
        super().__init__(episodes, reset_at_end)
        self.gamma = gamma

    def init_weights(self):
        if self.episode == 0:
            for arm in self.arms:
                setattr(arm, 'weight', 1)

    def _update_arm_weight(self, arm, reward):
        estimate = reward / self._arm_weight(arm)
        arm.weight *= math.exp(estimate * self.gamma / len(self.arms))

    def _arm_weight(self, arm):
        return (1.0 - self.gamma) * (arm.weight / self._w_sum()) + (self.gamma / len(self.arms))

    def _w_sum(self):
        return sum([arm.weight for arm in self.arms])

    def choose_arm(self):
        self.init_weights()
        w_dist = [self._arm_weight(arm) for arm in self.arms]
        chosen_arm = np.random.choice(self.arms, p=w_dist)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        arm = self.arm(name)
        self._update_arm_weight(arm, reward)
        arm.reward(reward)
