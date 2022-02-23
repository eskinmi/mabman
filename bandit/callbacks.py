import os
import json
from abc import ABC, abstractmethod
from typing import List


def checkin_params(path):
    file = open(F'{path}/agent_params.json', 'r')
    params = json.load(file)
    file.close()
    return params


def checkin_experiment(path):
    file = open(F'{path}/experiment.json', 'r')
    experiment_params = json.load(file)
    file.close()
    return experiment_params


def checkin_arms(path):
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

    def _checkout_experiment(self, experiment):
        with open(F'{self.path}/experiment.json', 'w') as f:
            json.dump(experiment.__dict__, f)

    def _checkout_arms(self, arms):
        data = {arm.name: arm.__dict__ for arm in arms}
        with open(F'{self.path}/arms.json', 'w') as f:
            json.dump(data, f)

    def _checkout_agent_params(self, agent):
        data = {k: v for k, v in agent.__dict__.items()
                if k not in ['experiment', 'arms', 'callbacks'] and
                not k.startswith('_')
                }
        with open(F'{self.path}/agent_params.json', 'w') as f:
            json.dump(data, f)

    def save(self, process):
        self._checkout_agent_params(process)
        self._checkout_experiment(process.experiment)
        self._checkout_arms(process.arms)


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

    def _save_history(self, hist):
        with open(F'{self.path}/hist.json', 'w') as f:
            json.dump(hist, f)

    def call(self, process):
        if process.experiment.is_completed:
            self._save_history(process.experiment.hist)


def callback(callbacks: List[CallBack], process):
    if callbacks:
        for cbk in callbacks:
            cbk.call(process)
