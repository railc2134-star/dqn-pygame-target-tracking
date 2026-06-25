# DQN Pygame Target Tracking

A Deep Q-Network (DQN) implementation built with PyTorch and Pygame.

The agent learns to move a green crosshair toward a randomly positioned target inside a custom simulated environment.

## Features

- Deep Q-Network implemented from scratch
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration
- Custom Pygame environment
- Real-time visualization

## Environment

The environment contains a green circle representing the agent and a red square representing the target.

At the beginning of each episode, both the agent and the target are placed at random positions.

The agent can perform four actions:

- Move up
- Move down
- Move left
- Move right

The reward is based on distance improvement.

A positive reward is given when the agent moves closer to the target.

A reward of 1 is given when the target is reached.

Episodes end when the target is reached or when 500 steps have been taken.

## Requirements

Install the required packages.

```bash
pip install torch pygame numpy
```

## Training

Set

```python
train = True
```

Run the script.

```bash
python dqn.py
```

The network parameters are periodically saved to

```text
nett.pth
```

## Evaluation

Set

```python
train = False
```

Run the script.

```bash
python dqn.py
```

The trained agent will control the crosshair and attempt to reach the target.
