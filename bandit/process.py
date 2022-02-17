
class Experiment:

    def __init__(self, n: int, episodes: int):
        self.episodes = episodes
        self.episode = 0
        self.n = n
        self.logs = []

    def next_episode(self):
        self.episode += 1

    def log(self, data):
        self.logs.append(data)

    @property
    def is_completed(self):
        return self.episodes - 1 == self.episode

    def __repr__(self):
        return F'(Experiment({n}), logs:{len(self.logs)})'


class ExperimentManager:

    def __init__(self, episodes: int, reset_at_end=False):
        self._experiments = []
        self.episodes = episodes
        self.reset_at_end = reset_at_end
        self.experiment_num = -1
        self.experiment = None
        self.stop = False
        self.new()

    @property
    def episode(self):
        return self.experiment.episode

    @property
    def experiments(self):
        return self._experiments + [self.experiment]

    def new(self):
        if self.experiment:
            self._experiments.append(self.experiment)
        self.experiment_num += 1
        self.experiment = Experiment(self.experiment_num, self.episodes)

    def update(self):
        if self.experiment.is_completed:
            if self.reset_at_end:
                self.new()
            else:
                self.stop = True
        else:
            self.experiment.next_episode()

    def log(self, data):
        self.experiment.log(data)


