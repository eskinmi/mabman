import math
import random
from typing import Union
from abc import ABC, abstractmethod
from bandit import ArmNotFoundException, RewardMissingException
from bandit import process
from bandit.arm import Arm


class Bandit(process.ExperimentManager, ABC):

    def __init__(self, episodes: int, reset_at_end: bool):
        super().__init__(episodes, reset_at_end)
        self.arms = []
        self.total_rewards = 0
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
    def reward_arm(self, arm: Arm, amount: Union[int, float]):
        pass

    def choose(self):
        if not self.stop and self.episode > self.episode_selected:
            return self.choose_arm()

    def reward(self, name: str, amount: Union[int, float] = 1):
        if self.episode_selected == self.episode:
            self.reward_arm(name, amount)
            self.update()
        else:
            raise RewardMissingException(self.episode)

    def arm(self, name:str):
        if res := list(filter(lambda x: x.name == name, self.arms)):
            return res[0]
        else:
            raise ArmNotFoundException(name)

    def add(self, name:str):
        self.arms.append(Arm(name))

    def remove(self, name:str):
        self.arms = [arm for arm in self.arms if arm.name != name]


class UpperConfidenceBoundBandit(Bandit):
    name = 'upper-confidence-bound-bandit'

    def __init__(self, episodes, reset_at_end, confidence:Union[int, float] = 2):
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
        self.total_rewards += amount
        self.episode_rewarded += 1


class EpsilonGreedyBandit(Bandit):
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
        self.total_rewards += amount
        self.episode_rewarded += 1


class EpsilonDecreasingBandit(Bandit):
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
        self.total_rewards += amount
        self.episode_rewarded += 1


class EpsilonFirstBandit(Bandit):
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
        self.total_rewards += amount
        self.episode_rewarded += 1


class BoltzmannBandit(Bandit):
    name = 'epsilon-first-bandit'

    def __init__(self, episodes, reset_at_end):
        super().__init__(episodes, reset_at_end)

    def choose_arm(self):
        denominator = sum([math.exp(a.mean_reward) for a in self.arms])
        chosen_arm = max(self.arms, key=lambda x: math.exp(x.mean_reward) / denominator)
        chosen_arm.select()
        self.episode_selected += 1
        return chosen_arm.name

    def reward_arm(self, name: str, amount):
        self.arm(name).reward(amount)
        self.total_rewards += amount
        self.episode_rewarded += 1
