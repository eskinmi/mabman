from bandit import utils
import json
from abc import ABC, abstractmethod
from typing import List


class WrongBanditCheckPointError(Exception):
    def __init__(self, name):
        self.message = F'checkpoint bandit module do not match current : {name}'
        super().__init__(self.message)


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
        utils.mkdirs(self.path)

    def save(self, process):
        arm_weights, experiment_params, agent_params = utils.agent_component_parts(process)
        utils.save_json(self.path + '/agent_params', agent_params)
        utils.save_json(self.path + '/experiment_params', experiment_params)
        utils.save_json(self.path + '/arm_weights', arm_weights)

    def load_component_weights(self):
        return (
            utils.read_json(self.path + '/arm_weights'),
            utils.read_json(self.path + '/experiment_params'),
            utils.read_json(self.path + '/agent_params'),

        )


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
        utils.mkdirs(self.path)

    def call(self, process):
        path = F'{self.path}/{str(process.experiment.experiment_id)}'
        utils.mkdirs(path)
        if process.experiment.is_completed:
            utils.save_json(path + '/hist.json', process.experiment.hist)


def callback(callbacks: List[CallBack], process):
    if callbacks:
        for cbk in callbacks:
            cbk.call(process)
