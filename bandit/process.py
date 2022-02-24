from bandit.callbacks import callback


class Experiment:

    def __init__(self, episodes: int = 1000):
        self.episodes = episodes
        self.episode = 0
        self.experiment_id = 0
        self.hist = {'actions': [], 'rewards': []}

    def next_episode(self):
        self.episode += 1

    def log(self, actions, rewards):
        self.hist['actions'].append(actions)
        self.hist['rewards'].append(rewards)

    @property
    def is_completed(self):
        return self.episodes - 1 == self.episode

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
        self.callbacks = callbacks
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
        self.experiment = Experiment(self.episodes)
        self.experiment.experiment_id += 1

    def proceed(self):
        callback(self.callbacks, self)
        if self.experiment.is_completed:
            if self.reset_at_end:
                self.new()
            else:
                self.stop = True
        else:
            self.experiment.next_episode()

    def add_episode_logs(self, name, reward, names):
        actions = [0] * len(names)
        actions[names.index(name)] = 1
        self.experiment.log(actions, reward)
