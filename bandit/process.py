from bandit.callbacks import apply_callbacks, _set_callbacks_list
from abc import abstractmethod
import itertools


class Experiment:

    def __init__(self, episodes: int = 1000):
        self.episodes = episodes
        self.episode = 0
        self.experiment_id = 0
        self.hist = []

    @property
    def is_completed(self):
        return self.episodes == self.episode + 1

    def summarize(self):
        arm_metrics = {}
        for i, g in itertools.groupby(sorted(self.hist), key=lambda x: x[0]):
            arm_metrics[i] = {}
            arm_metrics[i]['selections'] = (selections := sum(1 for _ in g))
            arm_metrics[i]['rewards'] = (rewards := sum(arr[1] for arr in g))
            arm_metrics[i]['mean_rewards'] = (rewards / selections if selections > 0 else 0)
        return arm_metrics

    def next_episode(self):
        self.episode += 1

    def log(self, **kwargs):
        self.hist.append(kwargs)

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
