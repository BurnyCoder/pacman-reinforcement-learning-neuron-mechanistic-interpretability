"""pacneuron: neuron-level interpretability and steering of reward-trained Ms. Pac-Man agents.

Global context: this package loads pretrained Sample Factory APPO agents (trained purely from
game reward, no imitation learning), replays them in the exact Atari preprocessing they were
trained with, records every neuron, labels neurons against emulator ground truth (RAM), and
tests labels causally by steering/ablating neurons in closed-loop play.
Entry point: `python -m pacneuron.pipeline --phase ...` (see pipeline.py).
"""
