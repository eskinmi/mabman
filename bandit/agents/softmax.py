from base import *


class SoftmaxBoltzmann(Agent):
    name = 'softmax-boltzmann-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 temperature
                 ):
        super().__init__(episodes, reset_at_end)
        self.temp = temperature

    def choose_arm(self):
        denominator = sum([math.exp(a.mean_reward / self.temp) for a in self.arms])
        probabilities = [math.exp(arm.mean_reward / self.temp) / denominator for arm in self.arms]
        chosen_arm = np.random.choice(self.arms, p=probabilities)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


class VDBE(Agent):
    name = 'epsilon-greedy-vdbe-bandit'

    def __init__(self,
                 episodes,
                 reset_at_end,
                 sigma,
                 init_epsilon=0.3
                 ):
        super().__init__(episodes, reset_at_end)
        self.sigma = sigma
        self.init_epsilon = init_epsilon
        self.prev_epsilon = self.init_epsilon
        self.agent_previous_mean_reward = self.agent_mean_reward

    @property
    def delta(self):
        if self.episode != 0:
            return 1 / self.episode
        else:
            return 1

    @property
    def action_value(self):
        prior = 1 - math.exp(-1 * abs(self.agent_mean_reward - self.agent_previous_mean_reward) / self.sigma)
        return (1 - prior) / (1 + prior)

    @property
    def epsilon(self):
        return self.delta * self.action_value + (1 - self.delta) * self.prev_epsilon

    def choose_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = max(self.arms, key=lambda x: x.mean_reward)
        else:
            chosen_arm = random.choice(self.arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.prev_epsilon = self.epsilon
        self.agent_previous_mean_reward = self.agent_mean_reward
        self.arm(name).reward(reward)
