from bandit import util
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
        util.mkdirs(self.path)

    def save_component_weights(self, process):
        util.save_json(self.path + '/weights.ckp', util.agent_component_weights(process))

    def load_component_weights(self):
        return util.read_json(self.path + '/weights.ckp')


class CheckPoint(CallBack):

    def __init__(self, in_every, path=None):
        super().__init__()
        self.ckp = CheckPointState(path)
        self.in_every = in_every
        raise NotImplementedError()

    def call(self, process):
        if process.experiment.episode != 0 and\
                process.experiment.episode % self.in_every == 0:
            self.ckp.save_component_weights(process)


class HistoryLogger(CallBack):

    def __init__(self, path=None):
        super().__init__()
        if path is None:
            self.path = './history'
        else:
            self.path = path
        util.mkdirs(self.path)

    def call(self, process):
        path = F'{self.path}/{str(process.experiment.experiment_id)}'
        util.mkdirs(path)
        if process.experiment.is_completed:
            util.save_json(path + '/hist.json', process.experiment.hist)


def apply_callbacks(callbacks: List[CallBack], process):
    if callbacks:
        for cbk in callbacks:
            cbk.call(process)


def _set_callbacks_list(callbacks: List[CallBack]):
    if callbacks is not None:
        if any(not isinstance(clb, CallBack) for clb in callbacks):
            raise ValueError('callbacks should be of callbacks.CallBack type.')
    return callbacks
