from base import *


class Hedge(Agent):
    name = 'hedge-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 temperature: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end)
        self.temperature = temperature

    def _threshold(self):
        return sum([math.exp(arm.rewards / self.temperature) for arm in self.arms])

    def choose_arm(self):
        th = self._threshold()
        z = random.random()
        chosen_arm = self.arms[-1]  # default
        p_sum = 0
        for arm in self.arms:
            p_sum += math.exp(arm.rewards / self.temperature) / th
            if p_sum > z:
                chosen_arm = arm
                break
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)
