import itertools
import random
from bandit.agents.base import Agent
from bandit.duels import DuelsMan, Duel
from bandit.arms import Arm
from typing import Optional, List, Set
import numpy as np
import math


class RUCB(Agent):
    name = 'relative-upper-confidence-bound-dueling-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 alpha: float = 0.75
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        if alpha <= 0.5:
            raise ValueError('alpha cannot be smaller than 0.5')
        self.alpha = alpha
        self.U = np.zeros((self.k, self.k))
        self.B, self.C = set(), set()
        self.duels = DuelsMan(self.arm_names)

    @staticmethod
    def hypothesized_best_arm_set(b: Set, c: Set):
        """
        B set represents the hypothesised best arms.
        It can be either empty or with a single element
        of champion, if the length of champion set (C)
        equals to 1.

        Parameters
        ----------
        b
        c

        Returns
        -------

        """
        if len(c) == 1:
            return c.copy()
        else:
            return b.intersection(c)

    def _champions_proba_dist(self):
        """
        Collection of champion selection
        probability distribution, given the
        condition that the |C| > 1.
            p(ac) = (if ac €; 0.5 ,otherwise; B, 2**|B| * |C - B|)

        Returns
        -------
        arm probability distributions from C.
        """
        return [
            0.5 if i in self.B else 1 / ((2 ** len(self.B)) * len(self.C - self.B))
            for i in self.C
        ]

    def compute_ucb(self):
        """
        Compute element-wise UCBs, in given W matrix with;
            U := [uij ] = W / W+WT + sqrt( a * lnt / W+WT )
            ; for each elem; x/0 <- 1.
        Returns
        -------
        UCB Matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ucb = (self.duels.W / (self.duels.W + self.duels.W.T)) + \
                  np.sqrt(self.alpha * (self.episode + 1) / (self.duels.W + self.duels.W.T))
            ucb = np.nan_to_num(ucb, nan=1.0)
            np.fill_diagonal(ucb, 0.5)
            return ucb

    def champions_set(self, u_ij):
        """
        Find initial champions set. Any arm that has  UCB >= 0.5
        :param u_ij: self.U (np.array(self.K, self.K))
        :return:
            C : Set[str]
        """
        return set(self.arm_names[i[0]] for i in np.argwhere(np.any(u_ij >= 0.5, axis=1)))

    def _find_champion(self):
        """
        Find champion from the C set.
        if |C| == 0, randomly select from the arms;
        otherwise random selection from a probability distribution.

        :return:
            arm name (str)
        """
        if len(self.C) == 0:
            return random.choice(self.arm_names)
        else:
            return np.random.choice(self.arm_names, p=self._champions_proba_dist())

    def _find_opponent(self, player):
        """
        Find opponent for a given champion.
        Looking at the win matrix of the champion (player),
        choose the highest winner. If there is a tie, choose
        one randomly. In this condition, opponent cannot be the
        same with the champion. Else, choose the highest winner
        :param player: arm name (str)
        :return:
            opponent arm name (str)
        """
        w_js = self.duels.W[self.duels.arm_index(player)]
        max_js = [j_max[0] for j_max in np.argwhere(w_js == w_js.max()).tolist()]
        if len(max_js) > 1:
            j = random.choice([j for j in max_js if j != max_js])
        else:
            j = max_js[0]
        return self.arm_names[j]

    def selection_policy(self):
        """
        applies selection policy.
        :return:
            Duel(a1, a2)
        """
        self.U = self.compute_ucb()
        self.C = self.champions_set(self.U)
        self.B = self.hypothesized_best_arm_set(self.B, self.C)
        player_one = self._find_champion()
        player_two = self._find_opponent(player_one)
        return self.duels.set(players=(player_one, player_two))

    def reward_policy(self, duel: Duel, winner: str):
        self.duels.step(duel, winner)


class REXP3(Agent):
    name = 'relative-exponential-weighting-dueling-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 gamma: float = 0.5
                 ):
        self._arm_vars_hook(weight=1, proba=0)
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.duels = DuelsMan(self.arm_names)
        self.gamma = gamma

    @property
    def weights_sum(self):
        return sum(arm.weight for arm in self.arms)

    def set_arm_weight(self, name: str, reward: int):
        """
        Set arm weight.

        Parameters
        ----------
        name: arm name
        reward: reward

        Returns
        -------
        None
        """
        arm = self.arm(name)
        arm.weight *= math.exp((self.gamma / len(self.arms)) * (reward / (2 * arm.proba)))

    def set_arm_proba(self, name: str):
        """
        Set arm selection probability with:
            [(1 - gamma) * w / €w ] + [gamma / N + K]
        Parameters
        ----------
        name : arm name

        Returns
        -------
        selection probability of arm
        """
        arm = self.arm(name)
        arm.proba = (1 - self.gamma) * arm.weight / self.weights_sum + self.gamma / (self.episode + 1) + self.k
        return arm.proba

    def selection_policy(self):
        p_dist = [self.set_arm_proba(name) for name in self.arm_names]
        player_one, player_two = np.random.choice(self.arms, size=2, p=p_dist).tolist()
        return self.duels.set(players=(player_one, player_two))

    def reward_policy(self, duel: Duel, winner: str):
        """
        reward policy:
            * set arm weights of winner and losers with relative reward.
                using the formula given in set_arm_weight.
            * update W matrix
        Parameters
        ----------
        duel: Duel
        winner: winner arm name (str)

        Returns
        -------
        None
        """
        winner, loser = duel.register_winner(winner)
        self.duels.step(duel, winner)
        if winner != loser:
            for name, reward in duel.rewards_dist():
                self.set_arm_weight(name, reward)


class CCB(Agent):
    name = 'copeland-confidence-bound-dueling-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 alpha: float = 0.75
                 ):
        raise NotImplementedError()
        self._arm_vars_hook(B=set())
        super().__init__(arms, episodes, reset_at_end, callbacks)
        self.alpha = alpha
        self.duels = DuelsMan(self.arm_names)
        self.U = np.zeros([self.k, self.k])
        self.L = np.zeros([self.k, self.k])
        self.C = set()
        self.Lc = self.k
        self.B = set(self.arm_names)

    def compute_confidence_bounds(self):
        """
        Compute element-wise UCBs and LCBs, in given W matrix with;
            U := [uij ] = W / W+WT +- sqrt( a * lnt / W+WT )
            ; for each elem; x/0 <- 1.
        Returns
        -------
        UCB Matrix.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = (self.duels.W / (self.duels.W + self.duels.W.T))
            ucb = mu + np.sqrt((self.alpha * np.log(self.episode + 1)) / (self.duels.W + self.duels.W.T))
            lcb = mu - np.sqrt((self.alpha * np.log(self.episode + 1)) / (self.duels.W + self.duels.W.T))
            ucb = np.nan_to_num(ucb, nan=1.0)
            lcb = np.nan_to_num(lcb, nan=0.0)
            np.fill_diagonal(ucb, 0.5)
            np.fill_diagonal(lcb, 0.5)
            return ucb, lcb

    def copeland(self, arr: np.ndarray, name: str):
        i = self.duels.arm_index(name)
        mask = np.ones(self.k, dtype=bool)
        mask[i] = False
        return np.argwhere((arr[i] >= 0.5) & mask).size

    def set_champions_set(self):
        max_jc = max(self.copeland(self.U, name) for name in self.arm_names)
        self.C = set(name for name in self.arm_names if self.copeland(self.U, name) == max_jc)

    def reset_params(self):
        for arm in self.arms:
            arm.B = set()
        self.B = set(self.arm_names)
        self.Lc = self.k

    def reset_disproven_hypotheses(self):
        pairs_check = [
            (self.duels.arm_index(arm.name), self.duels.arm_index(d))
            for arm in self.arms
            for d in arm.B
        ]
        rejected = any([self.L[pair] > 0.5 for pair in pairs_check])
        if rejected:
            self.reset_params()

    def remove_non_copeland_winners(self):
        if len(self.B) != 0:
            copeland_lower_js = [self.copeland(self.L, j) for j in self.arm_names]
            for i in self.B:
                copeland_i = self.copeland(self.U, i)
                if any([copeland_i < copeland_j for copeland_j in copeland_lower_js]):
                    self.B.discard(i)
                if len(self.arm(i).B) != self.Lc + 1:
                    self.arm(i).B = set(np.where(self.U[self.duels.arm_index(i)] < 0.5)[0])
        else:
            self.reset_params()

    def add_copeland_winners(self):
        confidence_bounds_eq = [
            (i, c_upper) for i in self.arm_names
            if (c_upper := self.copeland(self.U, i)) == self.copeland(self.L, i)
        ]
        if confidence_bounds_eq:
            for vals in confidence_bounds_eq:
                self.B.add(vals[0])
                self.arm(vals[0]).B = set()
                self.Lc = self.k - 1 - vals[1]
        for i, j in itertools.product(self.arm_names, self.arm_names):
            if i != j:
                b_len = len(self.arm(j).B)
                if b_len < self.Lc + 1:
                    self.arm(j).B = set()
                if b_len > self.Lc + 1:
                    potentials_sampled = np.random.choice(list(self.arm(j).B), size=self.Lc + 1, replace=False)
                    self.arm(j).B = potentials_sampled

    def type_3_pairs(self):
        pairs = []
        for i_name, j_name in itertools.product(self.arm_names, self.arm_names):
            i = self.duels.arm_index(i_name)
            j = self.duels.arm_index(j_name)
            if (self.U[i, j] == 0.5 or self.U[i, j] == 0.5) and (j_name in self.arm(i_name).B):
                pairs.append((i, j))
        return pairs

    def find_opponent(self, player, b_set):
        c = self.duels.arm_index(player)
        js = [
            self.duels.arm_index(name) for name in b_set
            if self.L[self.duels.arm_index(name), c] < 0.5
        ]
        argmax_u = max([self.U[j, c] for j in js])
        js_max = [(j, c) for j in js if self.U[j, c] == argmax_u]
        if len(js_max) > 1:
            return self.arm_names[random.choice([j for j, c in js_max if j != c])]
        else:
            return self.arm_names[js_max[0][0]]

    def selection_policy(self):
        self.U, self.L = self.compute_confidence_bounds()
        self.set_champions_set()
        self.reset_disproven_hypotheses()
        self.remove_non_copeland_winners()
        self.add_copeland_winners()
        if t3p := self.type_3_pairs() and random.random() <= 0.25:
            player_one, player_two = t3p[random.choice(range(len(t3p)))]
        else:
            if len(self.B.intersection(self.C)) != 0 and random.random() > 0.33:
                self.C = self.B.intersection(self.C)
            player_one = random.choice(list(self.C))
            if self.episode == 0 or random.random() > 0.5:
                b_i = set(self.arm_names)
            else:
                b_i = self.arm(player_one).B
            player_two = self.find_opponent(player_one, b_i)
        return self.duels.set(players=(player_one, player_two))

    def reward_policy(self, duel: Duel, winner: str):
        self.duels.step(duel, winner)