import math
import numpy as np
import random
from typing import Union
from abc import ABC, abstractmethod
from bandit import process
from bandit.arm import Arm, ArmNotFoundException, ArmAlreadyExistsException


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

