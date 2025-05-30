import pytest
import numpy as np
from .context import bandit
from bandit.agents.simple import (
    EpsilonGreedy, 
    EpsilonDecay,
    EpsilonFirst,
    UCB1Tuned
)
from bandit.arms import Arm

@pytest.fixture
def arms():
    return [
        Arm('a', p=0.7),
        Arm('b', p=0.3),
        Arm('c', p=0.5)
    ]

def test_epsilon_greedy_init():
    with pytest.raises(ValueError):
        EpsilonGreedy(epsilon=1.5)
    
    agent = EpsilonGreedy(epsilon=0.1)
    assert agent.epsilon == 0.1

def test_epsilon_greedy_selection(arms):
    np.random.seed(42)
    agent = EpsilonGreedy(arms=arms, epsilon=0)
    
    # With epsilon=0, should always choose best arm
    for _ in range(10):
        name = agent.choose()
        agent.reward(name, arms[0].draw())  # Arm 'a' has highest probability
    
    assert agent.arm('a').selections > agent.arm('b').selections

def test_ucb1_tuned_bounds(arms):
    agent = UCB1Tuned(arms=arms)
    
    # Initial bounds should be infinite
    for arm in arms:
        assert agent.calc_upper_bounds(arm) == float('inf')
    
    # After some selections, bounds should be finite
    name = agent.choose()
    agent.reward(name, 1)
    assert agent.calc_upper_bounds(agent.arm(name)) < float('inf')

def test_epsilon_first_phases(arms):
    agent = EpsilonFirst(arms=arms, episodes=100, epsilon=0.3)
    
    # Should be in exploration phase
    selections_before = [0] * len(arms)
    for _ in range(30):
        name = agent.choose()
        agent.reward(name, arms[0].draw())
        selections_before[ord(name) - ord('a')] += 1
    
    # Should have some variance in selections during exploration
    assert len(set(selections_before)) > 1