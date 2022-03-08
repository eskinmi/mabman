from typing import Union, Optional, List
from abc import ABC, abstractmethod

import numpy as np

from bandit import process
from bandit.states import EnvStates
from bandit.arms import Arm, ArmNotFoundException, ArmAlreadyExistsException
from bandit.callbacks import WrongBanditCheckPointError, CheckPointState


class MissingRewardException(Exception):
    def __init__(self, episode: int):
        self.message = F'round {episode} is not rewarded.'
        super().__init__(self.message)


class Agent(process.Process, ABC):

    def __init__(self,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 ):
        self.arms = []
        self.in_context = None
        self.init_arm_vars = dict()
        self.env = EnvStates()
        super().__init__(episodes, reset_at_end, callbacks)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def choose_arm(self, context=None):
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

    def _set_init_arm_attrs(self, **kwargs):
        self.init_arm_vars = kwargs

    def _update_attrs(self,  params: dict):
        self.__dict__.update(params)

    def add_arm(self, arm: Arm, overwrite: bool = False):
        if arm.name not in self.arm_names or overwrite:
            if overwrite:
                self.arms = list(filter(lambda x: x.name != arm.name, self.arms))
            [setattr(arm, k, v) for k, v in self.init_arm_vars.items()]
            self.arms.append(arm)
        else:
            raise ArmAlreadyExistsException(arm.name)

    def reset_arms(self):
        args = [(a.name, a.p) for a in self.arms]
        for arm_ags in args:
            self.add_arm(
                arm=Arm(arm_ags[0], arm_ags[1]),
                overwrite=True
                         )

    def deactivate_arm(self, name: str):
        self.arm(name).active = False

    def choose(self, context: Union[List, np.ndarray] = None):
        if not self.stop and self.episode_closed:
            return self.choose_arm(context)
        else:
            raise MissingRewardException(self.episode)

    def reward(self, name: str, reward: Union[int, float] = 1):
        if self.is_choice_made:
            self.reward_arm(name, reward)
            self.add_episode_logs(name, reward)
            self.proceed()
        else:
            raise MissingRewardException(self.episode)

    def arm(self, name: str):
        if name in self.arm_names:
            return self.arms[self.arm_names.index(name)]
        else:
            raise ArmNotFoundException(name)

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
