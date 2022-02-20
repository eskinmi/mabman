# mabman

This library is set to serve various implementations of multi armed bandit theory. Current implementations include:
* `UpperConfidenceBound`
* `EpsilonGreedy`
* `EpsilonDecay`
* `EpsilonFirst`
* `SoftmaxBoltzmann`
* `VDBE`
* `ThompsonSampling`

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
agent.reward(agent.choose(), reward=1)

```

## simulate

```python
from bandit import BernoulliArm, ThompsonSampling

agent = ThompsonSampling(100, False)
agent.add_arm(BernoulliArm('a', p=0.6))
agent.add_arm(BernoulliArm('b', p=0.1))
agent.add_arm(BernoulliArm('c', p=0.05))
agent.add_arm(BernoulliArm('d', p=0.05))
agent.add_arm(BernoulliArm('e', p=0.2))

while not agent.stop:
    name = agent.choose()
    reward = agent.arm(name).draw()
    agent.reward(name, reward)
```