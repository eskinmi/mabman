# mabman

This library is set to serve various implementations of multi armed bandit problem. The implementation includes some  intuitive 
additions to it, that the users will need when putting the MAB into practice. 
The package includes callbacks, which allows users to `log` and `checkpoint` their agent. Also, it allows users to run continuous / multiple
experiments without any code changes.  

Current agent implementations include:
* `UCB1`
* `UCB2`
* `Hedge`
* `EpsilonGreedy`
* `EpsilonDecay`
* `EpsilonFirst`
* `SoftmaxBoltzmann`
* `VDBE`
* `ThompsonSampling`
* `EXP3`
* `FPL`
* `LinUCB`

##  usage

```python
from bandit.agents.bayesian import VDBE
from bandit.arms import Arm
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
r = 1  # collect reward for arm here  
agent.reward(name, reward=r)
```
## simulation

```python
from bandit.agents.adversarial import EXP3
from bandit.arms import Arm

agent = EXP3(100, False)

agent.add_arm(Arm('a', p=0.6))
agent.add_arm(Arm('b', p=0.3))
agent.add_arm(Arm('c', p=0.4))

while not agent.stop:
    if name := agent.choose():
        r = agent.arm(name).draw()
        agent.reward(name, r)
```