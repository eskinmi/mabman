from base import *


class ThompsonSampling(Agent):
    name = 'thompson-sampling-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end
                 ):
        super().__init__(episodes, reset_at_end)

    def mk_draws(self):
        return [np.random.beta(arm.rewards + 1, arm.selections - arm.rewards + 1, size=1)
                for arm in self.arms
                ]

    def choose_arm(self):
        draws = self.mk_draws()
        chosen_arm = self.arms[draws.index(max(draws))]
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)

