import numpy as np
import math


def _check_duel_arr(arr):
    if not arr.shape[0] == arr.shape[1]:
        raise ValueError('array length and width should be equal.')


def copeland(arr: np.ndarray, i: str):
    _check_duel_arr(arr)
    k = arr.shape[0]
    mask = np.ones(k, dtype=bool)
    mask[i] = False
    return np.argwhere((arr[i] >= 0.5) & mask).size


def ucb(arm, episode, alpha):
    if arm.selections == 0:
        return 1e500
    else:
        return arm.mean_reward + (
            math.sqrt(alpha * math.log(episode + 1) / arm.selections)
        )


def lcb(arm, t, alpha):
    if arm.selections == 0:
        return 1e500
    else:
        return arm.mean_reward - (
            math.sqrt(alpha * math.log(t + 1) / arm.selections)
        )


def duel_ucb(wins: np.ndarray, t, alpha, nan=1.0):
    _check_duel_arr(wins)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (wins / (wins + wins.T)) + \
              np.sqrt(alpha * np.log(t + 1) / (wins + wins.T))
        u = np.nan_to_num(u, nan=nan)
        np.fill_diagonal(u, 0.5)
        return u


def duel_lcb(wins: np.ndarray, t, alpha, nan=1.0):
    _check_duel_arr(wins)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = (wins / (wins + wins.T)) - \
              np.sqrt(alpha * np.log(t + 1) / (wins + wins.T))
        u = np.nan_to_num(u, nan=nan)
        np.fill_diagonal(u, 0.5)
        return u


class SuccessiveSelector:

    def __init__(self):
        self.arm_name = None
        self.n = 0

    @property
    def in_recurrence(self):
        if self.arm_name is not None and self.n > 0:
            return True
        else:
            return False

    def set_recurrence(self, name, n):
        self.arm_name = name
        self.n = n

    def step(self):
        if self.in_recurrence:
            self.n -= 1
            return self.arm_name
        else:
            self.n = 0
            self.arm_name = None
