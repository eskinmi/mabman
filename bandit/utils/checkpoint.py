import pickle
import os
import shutil

CHECKPOINT_DIR = '../checkpoints/'
CHECKPOINT_1 = 'checkpoint_1.pkl'
CHECKPOINT_2 = 'checkpoint_2.pkl'


class CheckpointNotFoundException(Exception):
    def __init__(self):
        self.message = F'checkpoint not found!'
        super().__init__(self.message)


def _mk_dir():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)


def _backup():
    if os.path.exists(CHECKPOINT_DIR + CHECKPOINT_2):
        shutil.rmtree(CHECKPOINT_DIR + CHECKPOINT_2)
    if os.path.exists(CHECKPOINT_DIR + CHECKPOINT_1):
        shutil.copyfile(
            CHECKPOINT_DIR + CHECKPOINT_1,
            CHECKPOINT_DIR + CHECKPOINT_2
        )


def checkpoint(agent):
    with open(CHECKPOINT_DIR + CHECKPOINT_1, 'wb') as f:
        pickle.dump(agent, f)


def load():
    if os.path.exists((file := CHECKPOINT_DIR + CHECKPOINT_1)):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
    elif os.path.exists((file := CHECKPOINT_DIR + CHECKPOINT_2)):
        with open(file, 'rb') as f:
            agent = pickle.load(f)
    else:
        raise CheckpointNotFoundException()
    return agent


class CheckpointState:

    def __init__(self, in_every=None):
        self.in_every = in_every
        _mk_dir()

    def make(self, agent):
        if self.in_every and\
                agent.episode != 0 and\
                agent.episode % self.in_every == 0:
            _backup()
            checkpoint(agent)

