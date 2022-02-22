from base import *


class UCB1(Agent):
    name = 'upper-confidence-bound-1-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 confidence: Union[int, float] = 2
                 ):
        super().__init__(episodes, reset_at_end)
        self.confidence = confidence

    def calc_upper_bounds(self, arm):
        if arm.selections == 0:
            return 1e500
        else:
            return arm.mean_reward + (
                    self.confidence * math.sqrt(math.log(self.episode + 1) / arm.selections)
            )

    def choose_arm(self):
        chosen_arm = max(self.arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)
