import pytest
from .context import bandit
import random


class TestAgent(bandit.agents.base.Agent):
    name = 'test-bandit'

    def __init__(self,
                 episodes=100,
                 reset_at_end=False,
                 callbacks=None,
                 ):
        super().__init__(episodes, reset_at_end, callbacks)

    def choose_arm(self):
        chosen_arm = random.choice(self.active_arms)
        chosen_arm.select()
        return chosen_arm.name

    def reward_arm(self, name: str, reward):
        self.arm(name).reward(reward)


@pytest.fixture
def arms():
    return [
        bandit.arms.Arm('a', p=0.7),
        bandit.arms.Arm('b', p=0.3),
        bandit.arms.Arm('c', p=0.1),
        bandit.arms.Arm('d', p=0.2)
    ]


@pytest.fixture()
def base_agent(arms):
    agent = TestAgent(10, False)
    agent.arms = arms
    return agent


def test_choose(base_agent):
    name = base_agent.choose()
    assert base_agent.arm(name).selections == 1


def test_reward(base_agent):
    name = base_agent.choose()
    base_agent.reward(name, 1)
    assert base_agent.arm(name).rewards == 1
    with pytest.raises(bandit.agents.base.MissingRewardException):
        base_agent.reward(name, 1)


def test_add_arm(base_agent):
    base_agent.add_arm(bandit.arms.Arm('e'))
    assert base_agent.arms[-1].name == 'e'
    with pytest.raises(bandit.arms.ArmAlreadyExistsException):
        base_agent.add_arm(bandit.arms.Arm('a'))


def test_deactivate_arm(base_agent):
    base_agent.deactivate_arm('c')
    assert base_agent.arm('c').active is False
    assert 'c' not in [arm.name for arm in base_agent.active_arms]
