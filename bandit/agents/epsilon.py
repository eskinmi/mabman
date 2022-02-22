from bandit.agents.base import *
import random


class EpsilonGreedy(Agent):
    name = 'epsilon-greedy-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 epsilon: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class EpsilonDecay(Agent):
    name = 'epsilon-decreasing-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 epsilon: float = 0.5,
                 gamma: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon
        self.gamma = gamma

    def choose_arm(self):
        if random.random() > self.epsilon * (1-self.gamma)**self.episode:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class EpsilonFirst(Agent):
    name = 'epsilon-first-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 epsilon: float = 0.1
                 ):
        super().__init__(episodes, reset_at_end)
        self.epsilon = epsilon
        self.start_exploration = self.episode * (1-self.epsilon) - 1

    def choose_arm(self):
        if self.episode >= self.start_exploration:
            chosen_arm = random.choice(self.arms)
        else:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        chosen_arm.select()

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)
