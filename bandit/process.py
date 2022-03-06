from bandit.callbacks import apply_callbacks, _set_callbacks_list
from abc import abstractmethod


class Experiment:

    def __init__(self, episodes: int = 1000):
        self.episodes = episodes
        self.episode = 0
        self.experiment_id = 0
        self.hist = {}

    def next_episode(self):
        self.episode += 1

    def log(self, actions, rewards):
        self.hist['actions'].append(actions)
        self.hist['rewards'].append(rewards)

    @property
    def is_completed(self):
        return self.episodes == self.episode + 1

    def __repr__(self):
        return F'Experiment({self.experiment_id})'

    @classmethod
    def build(cls, params):
        experiment = cls()
        if params:
            experiment.__dict__.update(params)
        return experiment


class Process:

    def __init__(self,
                 episodes: int,
                 reset_at_end=False,
                 callbacks=None
                 ):
        self._experiments = []
        self.episodes = episodes
        self.reset_at_end = reset_at_end
        self.callbacks = _set_callbacks_list(callbacks)
        self.experiment = None
        self.experiment_num = 0
        self.stop = False
        self.new_experiment()

    @abstractmethod
    def reset_arms(self):
        pass

    @property
    def episode(self):
        return self.experiment.episode

    @property
    def experiments(self):
        return [*self._experiments, self.experiment]

    def new_experiment(self):
        if self.experiment:
            self._experiments.append(self.experiment)
        self.experiment = Experiment(self.episodes)
        self.experiment_num += 1
        self.experiment.experiment_id = self.experiment_num
        self.reset_arms()

    def proceed(self):
        apply_callbacks(self.callbacks, self)
        if self.experiment.is_completed:
            if self.reset_at_end:
                self.new_experiment()
            else:
                self.stop = True
        else:
            self.experiment.next_episode()

    def add_episode_logs(self, name, reward, names):
        # log here only the name of selected arm, not in array
        actions = [0] * len(names)
        actions[names.index(name)] = 1
        self.experiment.log(actions, reward)
