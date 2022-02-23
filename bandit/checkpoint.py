from bandit.agents import Agent
from bandit.arms import Arm
from bandit.process import Experiment
import os
import json


def _checkout_experiment(path, experiment):
    with open(F'{path}/experiment.json', 'w') as f:
        json.dump(experiment.__dict__, f)


def _checkout_arms(path, arms):
    data = {arm.name: arm.__dict__ for arm in arms}
    with open(F'{path}/arms.json', 'w') as f:
        json.dump(data, f)


def _checkout_agent_params(path, agent):
    data = {
        'name': agent.__class__.name,
        'params': {k: v for k, v in agent.__dict__.items()
                   if k not in ['experiment', 'arms'] and
                   not k.startswith('_')
                   }
    }
    with open(F'{path}/agent_params.json', 'w') as f:
        json.dump(data, f)


def _checkin_agent_params(path):
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


def _agent_names():
    return {c.name: c for c in Agent.__subclasses__()}


class CheckPointState:

    def __init__(self, path=None):
        if path is None:
            self.path = './checkpoints'
        else:
            self.path = path
        self._mkdirs()

    def _mkdirs(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, process: Agent):
        """
        Check-outing or checkpointing process for the
        agent. This allows users to load the agent back
        from the latest state and keep training.
        Only checkouts the current experiment.
        The previous experiments logged in process.Process
        should be saved (if needed) by the user.

        :param process: process.Process to checkpoint
        :return: None
        """
        _checkout_agent_params(self.path, process)
        _checkout_experiment(self.path, process.experiment)
        _checkout_arms(self.path, process.arms)

    def load(self):
        params = _checkin_agent_params(self.path)
        experiment_params = _checkin_experiment(self.path)
        experiment = Experiment()
        experiment.__dict__.update(experiment_params)
        arms_params = _checkin_arms(self.path)
        arms = [Arm(name=k).__dict__.update(v) for k, v in arms_params.items()]
        agent_cls = _agent_names().pop(params['name'])()
        agent_cls.__dict__.update(params['params'])
        agent_cls.experiment = experiment
        agent_cls.arms = arms
        return agent_cls

