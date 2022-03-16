import numpy as np


class StatesSecretary:

    def __init__(self):
        self.past_states = []
        self.state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state is not None:
            self._state = np.reshape(state, (-1, 1))
        else:
            self._state = None

    @property
    def last_state(self):
        if self.past_states:
            return self.past_states[-1]

    def instate(self, state):
        self.state = state

    def add_hist(self):
        if self.state is not None:
            self.past_states.append(self.state)

    def __enter__(self):
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.add_hist()
        self.state = None
