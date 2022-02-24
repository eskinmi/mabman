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

agent = VDBE(100, False, 0.5, 0.3)
agent.add_arm(Arm('a'))
agent.add_arm(Arm('b'))
agent.add_arm(Arm('c'))
agent.add_arm(Arm('d'))
agent.add_arm(Arm('e'))

# process
name = agent.choose()
rew = None # collect reward for arm here  
agent.reward(name, reward=rew)

```

## simulate

```python
from bandit import VDBE, BernoulliArm
from bandit.callbacks import HistoryLogger, CheckPoint
callbacks = [HistoryLogger(), CheckPoint(50)]
agent = VDBE(100, False, callbacks=callbacks)
agent.add_arm(BernoulliArm('a', p=0.6))
agent.add_arm(BernoulliArm('b', p=0.3))
agent.add_arm(BernoulliArm('c', p=0.4))
while not agent.stop:
    if name := agent.choose():
        amt = agent.arm(name).draw()
        agent.reward(name, amt)
```