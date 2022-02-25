import pytest
from .context import bandit


@pytest.fixture()
def experiment():
    return bandit.process.Experiment(10)


@pytest.fixture()
def process_iter():
    return bandit.process.Process(20, True)


@pytest.fixture()
def process_non_iter():
    return bandit.process.Process(20, False)


def test_experiment_completion(experiment):
    experiment.episode = 9
    assert experiment.is_completed is True


def test_process_new_experiment(process_iter):
    process_iter.new_experiment()
    assert len(process_iter.experiments) == 2
    assert process_iter.experiments[0].experiment_id == 1
    assert process_iter.experiments[1].experiment_id == 2


def test_process_stop(process_non_iter):
    for _ in range(22):
        process_non_iter.proceed()
    assert process_non_iter.stop is True
    assert len(process_non_iter.experiments) == 1
    assert process_non_iter.experiment_num == 1


def test_process_continuation(process_iter):
    for _ in range(21):
        process_iter.proceed()
    assert process_iter.stop is False
    assert process_iter.episode == 1
