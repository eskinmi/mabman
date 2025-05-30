__all__ = [
    'EpsilonGreedy',
    'EpsilonDecay',
    'EpsilonFirst',
    'Hedge',
    'SoftmaxBoltzmann',
    'ThompsonSampling',
    'UCB1',
    'UCB1Tuned',
    'VDBE'
]

import random
import math
import numpy as np
from typing import Optional, Union, List
import logging
from bandit.utils.policy import SuccessiveSelector, ucb
from bandit.agents.base import Agent
from bandit.arms import Arm

logger = logging.getLogger(__name__)

class EpsilonGreedy(Agent):
    """Epsilon Greedy bandit algorithm implementation.
    
    Implements the epsilon-greedy strategy where the agent explores with 
    probability epsilon and exploits with probability 1-epsilon.
    """
    name = 'epsilon-greedy-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.1
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
        self.epsilon = epsilon
        logger.info(f"Initialized {self.name} with epsilon={epsilon}")

    def selection_policy(self) -> str:
        if random.random() > self.epsilon:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
            logger.debug("Exploiting best arm")
        else:
            chosen_arm = random.choice(self.active_arms)
            logger.debug("Exploring random arm")
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)
        logger.debug(f"Arm {name} received reward {reward}")


class EpsilonDecay(Agent):
    """Epsilon Decreasing bandit algorithm implementation.
    
    Implements the epsilon-decreasing strategy where exploration probability
    decreases over time.
    """
    name = 'epsilon-decreasing-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.5,
                 gamma: float = 0.1,
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")
        self.epsilon = epsilon
        self.gamma = gamma
        logger.info(f"Initialized {self.name} with epsilon={epsilon}, gamma={gamma}")

    def selection_policy(self) -> str:
        current_epsilon = self.epsilon * (1 - self.gamma)**self.episode
        if random.random() > current_epsilon:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
            logger.debug("Exploiting best arm")
        else:
            chosen_arm = random.choice(self.active_arms)
            logger.debug("Exploring random arm")
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)
        logger.debug(f"Arm {name} received reward {reward}")


class EpsilonFirst(Agent):
    """Epsilon First bandit algorithm implementation.
    
    Implements the epsilon-first strategy where exploration happens first,
    followed by exploitation.
    """
    name = 'epsilon-first-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 epsilon: float = 0.1,
                 ):
        super().__init__(arms, episodes, reset_at_end, callbacks)
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1")
        self.epsilon = epsilon
        self.start_exploration = int(self.episodes * (1-self.epsilon))
        logger.info(f"Initialized {self.name} with epsilon={epsilon}")

    def selection_policy(self) -> str:
        if self.episode >= self.start_exploration:
            chosen_arm = max(self.active_arms, key=lambda x: x.mean_reward)
            logger.debug("Exploiting best arm")
        else:
            chosen_arm = random.choice(self.active_arms)
            logger.debug("Exploring random arm")
        chosen_arm.select()
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        self.arm(name).reward(reward)
        logger.debug(f"Arm {name} received reward {reward}")


class UCB1Tuned(Agent):
    """UCB1-Tuned bandit algorithm implementation.
    
    Implements the UCB1-Tuned algorithm which uses variance estimates
    to improve exploration.
    """
    name = 'upper-confidence-bound-1-tuned-bandit'

    def __init__(self,
                 arms: Optional[List[Arm]] = None,
                 episodes: int = 100,
                 reset_at_end: bool = False,
                 callbacks: Optional[list] = None,
                 confidence: Union[int, float] = 2
                 ):
        self._arm_vars_hook(rewards_sq=0)
        self.confidence = confidence
        super().__init__(arms, episodes, reset_at_end, callbacks)
        logger.info(f"Initialized {self.name} with confidence={confidence}")

    def arm_variance_ub(self, arm: Arm) -> float:
        if arm.selections == 0:
            return float('inf')
        return (
            arm.rewards_sq / arm.selections - (arm.mean_reward ** 2) +
            math.sqrt(self.confidence * math.log(self.episode + 1) / arm.selections)
        )

    def calc_upper_bounds(self, arm: Arm) -> float:
        if arm.selections == 0:
            return float('inf')
        return arm.mean_reward + math.sqrt(
            min(0.25, self.arm_variance_ub(arm)) * 
            math.log(self.episode + 1) / arm.selections
        )

    def selection_policy(self) -> str:
        chosen_arm = max(self.active_arms, key=lambda x: self.calc_upper_bounds(x))
        chosen_arm.select()
        logger.debug(f"Selected arm {chosen_arm.name} based on UCB1-Tuned bounds")
        return chosen_arm.name

    def reward_policy(self, name: str, reward: Union[int, float]):
        arm = self.arm(name)
        arm.reward(reward)
        arm.rewards_sq += reward ** 2
        logger.debug(f"Arm {name} received reward {reward}")