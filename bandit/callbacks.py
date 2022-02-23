import os
import json
from abc import ABC, abstractmethod
from typing import List


def _checkout_experiment(path, experiment):
    with open(F'{path}/experiment.json', 'w') as f:
        json.dump(experiment.__dict__, f)


def _checkout_arms(path, arms):
    data = {arm.name: arm.__dict__ for arm in arms}
    with open(F'{path}/arms.json', 'w') as f:
        json.dump(data, f)


def _checkout_agent_params(path, agent):
    data = {k: v for k, v in agent.__dict__.items()
            if k not in ['experiment', 'arms', 'callbacks'] and
            not k.startswith('_')
            }
    with open(F'{path}/agent_params.json', 'w') as f:
        json.dump(data, f)


def _checkin_params(path):
    file = open(F'{path}/agent_params.json', 'r')
    params = json.load(file)
    file.close()
    return params


def _checkin_experiment(path):
    file = open(F'{path}/experiment.json', 'r')
    experiment_params = json.load(file)
    file.close()
    return experiment_params


def _checkin_arms(path):
    file = open(F'{path}/arms.json', 'r')
    arms_params = json.load(file)
    file.close()
    return arms_params


def _mkdirs(path):
    if not os.path.exists(path):
            os.makedirs(path)

class CallBack(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def call(self, process):
        pass


class CheckPointState:

    def __init__(self, path=None):
        super().__init__()
        if path is None:
            self.path = './checkpoints'
        else:
            self.path = path
        _mkdirs(self.path)

    def save(self, process):
        """
        Check-outing or checkpointing process for the
        agent. This allows users to load the agent back
        from the latest state and keep training.
        Only checkouts the current experiment.
        The previous experiments logged in process.Process
        should be saved (if needed) by the user.
        """
        _checkout_agent_params(self.path, process)
        _checkout_experiment(self.path, process.experiment)
        _checkout_arms(self.path, process.arms)

    def load(self, agent):
        from bandit.arms import Arm
        from bandit.process import Experiment
        params = _checkin_params(self.path)
        experiment_params = _checkin_experiment(self.path)
        experiment = Experiment()
        experiment.__dict__.update(experiment_params)
        arms_params = _checkin_arms(self.path)
        arms = [Arm(name=k).__dict__.update(v) for k, v in arms_params.items()]
        agent_cls = agent()
        agent_cls.__dict__.update(params['params'])
        agent_cls.experiment = experiment
        agent_cls.arms = arms
        return agent_cls


class CheckPoint(CallBack):

    def __init__(self, in_every, path=None):
        super().__init__()
        self.ckp = CheckPointState(path)
        self.in_every = in_every

    def call(self, process):
        if process.experiment.episode != 0 and\
                process.experiment.episode % self.in_every == 0:
            self.ckp.save(process)


class HistoryLogger(CallBack):

    def __init__(self, path=None):
        super().__init__()
        if path is None:
            self.path = './history'
        else:
            self.path = path
        _mkdirs(self.path)

    def _log_history(self, hist):
        with open(F'{self.path}/hist.json', 'w') as f:
            json.dump(hist, f)

    def call(self, process):
        if process.experiment.is_completed:
            self._log_history(process.experiment.hist)


def callback(callbacks: List[CallBack], process):
    if callbacks:
        for cbk in callbacks:
            cbk.call(process)
