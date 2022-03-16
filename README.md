# mabman

This library is set to serve various implementations of multi armed bandit problem. The implementation includes some  intuitive 
additions to it, that the users will need when putting the MAB into practice. 
The package includes callbacks, which allows users to `log` and `checkpoint` their agent. Also, it allows users to run continuous / multiple
experiments without any code changes.

## simulation

```python
from bandit.agents.adversarial import EXP3
from bandit.arms import Arm
from bandit.callbacks import HistoryLogger, CheckPoint

callbacks = [
    HistoryLogger(),
    CheckPoint(in_every=50)
]
arms = [Arm('a', p=0.6), Arm('b', p=0.3), Arm('c', p=0.4)]
agent = EXP3(arms=arms, episodes=1000, callbacks=callbacks)
while not agent.stop:
    if name := agent.choose():
        r = agent.arm(name).draw()
        agent.reward(name, r)
```

