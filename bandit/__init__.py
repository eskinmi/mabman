from bandit.bandits import *
from bandit.arm import Arm


class BanditNotFoundException(Exception):
    def __init__(self, name):
        self.message = F'bandit({name}) not found!'
        super().__init__(self.message)


class ArmNotFoundException(Exception):
    def __init__(self, name):
        self.message = F'arm({name}) not found!'
        super().__init__(self.message)


class RewardMissingException(Exception):
    def __init__(self, episode: int):
        self.message = F'round {episode} is not rewarded.'
        super().__init__(self.message)