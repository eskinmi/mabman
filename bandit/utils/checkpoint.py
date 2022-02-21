import pickle
import os
import shutil

class CheckpointNotFoundException(Exception):
    def __init__(self):
        self.message = F'checkpoint not found!'
        super().__init__(self.message)


def _mk_dir(path: str = '../checkpoints/'):
    if not os.path.exists(path):
        os.makedirs(path)


def _backup(path: str = '../checkpoints/'):
    if os.path.exists(path + 'checkpoint_2.pkl'):
        shutil.rmtree(path + 'checkpoint_2.pkl')
    if os.path.exists(path + 'checkpoint_1.pkl'):
        shutil.copyfile(
            path + 'checkpoint_1.pkl',
            path + 'checkpoint_2.pkl'
        )


def checkpoint(path: str = '../checkpoints/', agent):
    with open(path + 'checkpoint_1.pkl', 'wb') as f:
        pickle.dump(agent, f)


def load(path):
    if os.path.exists((file := path + 'checkpoint_1.pkl')):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
    elif os.path.exists((file := path + 'checkpoint_2.pkl')):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
    else:
        raise CheckpointNotFoundException()
    return agent


class CheckpointState:

    def __init__(self, path: str = '../checkpoints/', in_every=None):
        self.in_every = in_every
        self.path = path
        _mk_dir()

    def make(self, agent):
        if self.in_every and\
                agent.episode != 0 and\
                agent.episode % self.in_every == 0:
            _backup(self.path)
            checkpoint(self.path, agent)

