import math
import numpy as np
import random
from typing import Union
from abc import ABC, abstractmethod
from bandit import process
from bandit.arm import Arm, ArmNotFoundException


class MissingRewardException(Exception):
    def __init__(self, episode: int):
        self.message = F'round {episode} is not rewarded.'
        super().__init__(self.message)


class Bandit(process.Process, ABC):

    def __init__(self, episodes: int, reset_at_end: bool):
        super().__init__(episodes, reset_at_end)
        self.arms = []
        self.episode_rewarded = -1
        self.episode_selected = -1

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def choose_arm(self):
        pass

    @abstractmethod
    def reward_arm(self, name: str, amount: Union[int, float]):
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
    def episode_log(self):
        return [arm.selections for arm in self.arms], [arm.rewards for arm in self.arms]

    def choose(self):
        if not self.stop and self.episode > self.episode_selected:
            return self.choose_arm()

    def reward(self, name: str, amount: Union[int, float] = 1):
        if self.episode_selected == self.episode:
            self.reward_arm(name, amount)
            self.proceed()
        else:
            raise MissingRewardException(self.episode)

    def arm(self, name: str):
        if res := list(filter(lambda x: x.name == name, self.arms)):
            return res[0]
        else:
            raise ArmNotFoundException(name)

    def add(self, name: str):
        self.arms.append(Arm(name))

    def remove(self, name: str):
        self.arms = [arm for arm in self.arms if arm.name != name]


class UpperConfidenceBound(Bandit):
    name = 'upper-confidence-bound-bandit'

    def __init__(self, episodes, reset_at_end, confidence: Union[int, float] = 2):
        super().__init__(episodes, reset_at_end)
        self.confidence = confidence

    def choose_arm(self):
        chosen_arm = None
        max_upper_bound = 0
        for arm in self.arms:
            if arm.selections > 0:
                delta_i = math.sqrt(self.confidence * math.log(self.episode+1) / arm.selections)
                upper_bound = arm.mean_reward + delta_i
            else:
                upper_bound = 1e500
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                chosen_arm = arm
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.episode_rewarded += 1


class EpsilonGreedy(Bandit):
    name = 'epsilon-greedy-bandit'

    def __init__(self, episodes, reset_at_end, epsilon: float = 0.1):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.episode_rewarded += 1


class EpsilonDecay(Bandit):
    name = 'epsilon-decreasing-bandit'

    def __init__(self, episodes, reset_at_end, epsilon: float = 0.5, gamma: float = 0.1):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_arm(self):
        if random.random() > self.epsilon * (1-self.gamma)**self.episode:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.episode_rewarded += 1


class EpsilonFirst(Bandit):
    name = 'epsilon-first-bandit'

    def __init__(self, episodes, reset_at_end, epsilon: float = 0.1):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon
        self.start_exploration = self.episode * (1-self.epsilon) - 1

    def choose_arm(self):
        if self.episode >= self.start_exploration:
            chosen_arm = random.choice(self.arms)
        else:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        chosen_arm.select()
        self.episode_selected += 1

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.episode_rewarded += 1


class SoftmaxBoltzmann(Bandit):
    name = 'softmax-boltzmann-bandit'

    def __init__(self, episodes, reset_at_end, temperature):
        super().__init__(episodes, reset_at_end)
        self.temp = temperature

    def choose_arm(self):
        denominator = sum([math.exp(a.mean_reward / self.temp) for a in self.arms])
        probabilities = [math.exp(arm.mean_reward / self.temp) / denominator for arm in self.arms]
        chosen_arm = np.random.choice(self.arms, p=probabilities)
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.episode_rewarded += 1


class EpsilonGreedyVDBE(Bandit):
    name = 'epsilon-greedy-vdbe-bandit'

    def __init__(self, episodes, reset_at_end, sigma, init_epsilon=0.3):
        super().__init__(episodes, reset_at_end)
        self.sigma = sigma
        self.init_epsilon = init_epsilon
        self.prev_epsilon = self.init_epsilon
        self.agent_previous_mean_reward = self.agent_mean_reward

    @property
    def delta(self):
        if self.episode != 0:
            return 1 / self.episode
        else:
            return 1

    @property
    def action_value(self):
        prior = 1 - math.exp(-1 * abs(self.agent_mean_reward - self.agent_previous_mean_reward) / self.sigma)
        return (1 - prior) / (1 + prior)

    @property
    def epsilon(self):
        return self.delta * self.action_value + (1 - self.delta) * self.prev_epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.prev_epsilon = self.epsilon
        self.agent_previous_mean_reward = self.agent_mean_reward
        self.arm(name).reward(amount)
        self.episode_rewarded += 1


class ThompsonSampling(Bandit):
    name = 'thompson-sampling-bandit'

    def __init__(self, episodes, reset_at_end):
        super().__init__(episodes, reset_at_end)

    def mk_draws(self):
        return [np.random.beta(arm.rewards + 1, arm.selections - arm.rewards, size=1)
                for arm in self.arms
                ]

    def choose_arm(self):
        draws = self.mk_draws()
        chosen_arm = self.arms[draws.index(max(draws))]
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.episode_rewarded += 1
