from typing import Optional, List
from abc import ABC, abstractmethod

from bandit import process
from bandit.arms import Arm, ArmNotFoundException, ArmAlreadyExistsException
from bandit.callbacks import WrongBanditCheckPointError, CheckPointState


class Agent(process.Process, ABC):

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 ):
        self.arms = []
        if arms is not None:
            [self.add_arm(arm) for arm in arms]
        super().__init__(episodes, reset_at_end, callbacks)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def selection_policy(self, *args, **kwargs):
        pass

    @abstractmethod
    def reward_policy(self, *args, **kwargs):
        pass

    @property
    def k(self):
        return len(self.arms)

    @property
    def active_arms(self):
        return list(filter(lambda x: x.active, self.arms))

    @property
    def rewards(self):
        return sum(arm.rewards for arm in self.arms)

    @property
    def agent_mean_reward(self):
        if self.episode > 0:
            return self.rewards / self.episode
        else:
            return 0

    @property
    def arm_names(self):
        return [arm.name for arm in self.arms]

    def _arm_vars_hook(self, **kwargs):
        self.init_arm_vars = kwargs

    def _update_attrs(self,  params: dict):
        self.__dict__.update(params)

    def add_arm(self, arm: Arm, overwrite: bool = False):
        if arm.name not in self.arm_names or overwrite:
            if overwrite:
                self.arms = list(filter(lambda x: x.name != arm.name, self.arms))
            if hasattr(self, 'init_arm_vars'):
                for k, v in self.init_arm_vars.items():
                    if not hasattr(arm, k):
                        setattr(arm, k, v)
                    else:
                        raise ValueError(f'attribute {k} already exists in arm, please rename.!')
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

    def choose(self, *args, **kwargs):
        if not self.stop:
            return self.selection_policy(*args, **kwargs)

    def reward(self, *args, **kwargs):
        self.reward_policy(*args, **kwargs)
        self.experiment.log()
        self.proceed()

    def arm(self, name: str):
        if name in self.arm_names:
            return self.arms[self.arm_names.index(name)]
        else:
            raise ArmNotFoundException(name)

    def overlay_weights(self, path):
        ckp = CheckPointState(path)
        weights = ckp.load_component_weights()
        if self.name == weights['agent']['name']:
            self.arms = [
                Arm.build(arm_weights['name'], arm_weights['weights'])
                for arm_weights in weights['arm']
            ]
            self.experiment = process.Experiment.build(weights['experiment'])
            self._update_attrs(weights['agent']['params'])
        else:
            raise WrongBanditCheckPointError(weights['agent']['name'])
