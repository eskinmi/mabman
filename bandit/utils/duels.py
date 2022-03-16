from typing import List, Tuple
import numpy as np


class WinnerNotRegistered(Exception):
    def __init__(self, message: str = 'winner not registered for duel.'):
        self.message = message
        super().__init__(self.message)


class Duel:

    def __init__(self, opponents: Tuple[str]):
        self.winner = None
        self.opponents = opponents
        self.duel_id = 0

    @property
    def loser(self):
        if self.winner is not None:
            return player if (player := self.opponents[0]) != self.winner else self.opponents[1]

    @property
    def is_rewarded(self):
        return self.winner is not None

    def register_winner(self, winner):
        if winner is not None:
            self.winner = winner
        else:
            raise ValueError('winner cannot be None')
        return self.winner, self.loser

    def rewards_dist(self):
        return [(self.winner, 1), (self.loser, -1)]

    def __repr__(self):
        return F'Duel({self.opponents})'


class DuelsMan:

    def __init__(self, players: List[str]):
        self.players = players
        self.K = len(self.players)
        self.duels = []
        self.duel_num = 0
        self.W = np.zeros((self.K, self.K))

    def set(self, players):
        self.duels.append([players])
        self.duel_num += 1
        duel = Duel(players)
        duel.duel_id = self.duel_num
        return duel

    def arm_index(self, name):
        return self.players.index(name)

    def step(self, duel: Duel, winner: str):
        winner, loser = duel.register_winner(winner)
        self.W[self.arm_index(winner), self.arm_index(loser)] += 1
