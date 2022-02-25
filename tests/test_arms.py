import pytest
from .context import bandit


@pytest.fixture
def bernoulli_arm():
    return bandit.arms.Arm('a', p=0.7)


@pytest.fixture
def arm():
    return bandit.arms.Arm('b')


@pytest.fixture()
def arm_weights():
    return {
        'p': 0.3,
        'selections': 10,
        'rewards': 5,
    }


def test_update_mean_reward(arm):
    arm.select()
    arm.reward(1)
    assert arm.mean_reward == 1
    arm.select()
    arm.reward(0)
    assert arm.mean_reward == 0.5


def test_arm_draw(bernoulli_arm):
    bernoulli_arm.p = None
    with pytest.raises(ValueError):
        bernoulli_arm.draw()


def test_arm_build(arm_weights):
    arm = bandit.arms.Arm.build('x', arm_weights)
    assert arm.p == 0.3
    assert arm.selections == 10
    assert arm.rewards == 5
    assert arm.mean_reward == 0.0
