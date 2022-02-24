# mabman

This library is set to serve various implementations of multi armed bandit theory. Current implementations include:
* `UCB1`
* `Hedge`
* `EpsilonGreedy`
* `EpsilonDecay`
* `EpsilonFirst`
* `SoftmaxBoltzmann`
* `VDBE`
* `ThompsonSampling`
* `EXP3`

## implentation

```python
from bandit import Arm, VDBE
from bandit.callbacks import HistoryLogger, CheckPoint

callbacks = [
    HistoryLogger(),
    CheckPoint(in_every=50)
]
agent = VDBE(100, False, sigma=.5, init_epsilon=0.3, callbacks=callbacks)

agent.add_arm(Arm('a'))
agent.add_arm(Arm('b'))
agent.add_arm(Arm('c'))
agent.add_arm(Arm('d'))
agent.add_arm(Arm('e'))

# process
name = agent.choose()
rew = 1 # collect reward for arm here  
agent.reward(name, reward=rew)
```

## simulate

```python
from bandit import VDBE, Arm
agent = VDBE(100, False)

agent.add_arm(Arm('a', p=0.6))
agent.add_arm(Arm('b', p=0.3))
agent.add_arm(Arm('c', p=0.4))

while not agent.stop:
    if name := agent.choose():
        amt = agent.arm(name).draw()
        agent.reward(name, amt)
```