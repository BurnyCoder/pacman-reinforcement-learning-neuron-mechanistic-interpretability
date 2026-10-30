# Experiment 1: Fidelity gate (does our code run the pretrained agent exactly as it was trained?)

## Why
Sample Factory cannot run on native Windows. Without a check, a re-implementation could silently differ from the
agent that was actually trained, and every neuron label would then describe a different network. Two common
pitfalls:
- gymnasium ≥ 1.0 changed its grayscale wrapper from `cv2.cvtColor` to NumPy weights
  (`0.2125/0.7154/0.0721`, truncating). This shifts input pixels by up to a few gray levels.
- The observation normaliser (running mean/std, float64 buffers cast to float32, clip ±5) must be applied exactly.

## Method
- **Reference.** `scripts/sf_golden_wsl.sh` builds the training-era stack in WSL: sample-factory 2.0.3, gym 0.26.2,
  ale-py 0.8.1, torch 1.13.1. The pretrained runs predate the 2.0.0 release (Oct vs Dec 2022), so 2.0.3 is the
  closest released stack. With Sample Factory's own `make_atari_env` and `create_actor_critic`, the script plays a
  fixed random action sequence (env seed 123) and records observations, logits and values.
- **Our implementation.**
  - `src/pacneuron/env.py`: wrappers ported from `sample_factory/envs/env_wrappers.py`, plus gym-0.26-style
    resize/grayscale/frame-stack using cv2.
  - `src/pacneuron/model.py`: plain PyTorch, loaded with a strict key and shape map.
  - Both replay the same actions on Windows with gymnasium 1.4 / ale-py 0.12.1.

## Result
<!-- BEGIN:fidelity -->
| check | value |
|---|---|
| golden trajectory steps (incl. 2 life losses and game over) | 446 |
| steps with any observation pixel mismatch | 0 |
| max abs. logit difference | 1.1e-05 |
| max abs. value difference | 4.8e-07 |
| reference stack | sample-factory 2.0.3, gym 0.26.2, ale-py 0.8.1, torch 1.13.1 (WSL) |
<!-- END:fidelity -->

The gate is enforced by `tests/test_fidelity.py`: observations must be bit-exact; logits and values within
1e-5 (absolute and relative tolerance).

## Learnings
- ale-py 0.12 still registers `MsPacmanNoFrameskip-v4` with `max_num_frames_per_episode=108000`, the same as ale-py
  0.8.1 (checked in the ALE 0.8.1 source). The episode limit therefore matches training.
- Rewriting the 5 Atari wrappers by hand (about 100 lines) proved safer than relying on any library version's
  wrapper semantics.
- A score check alone would be a weak gate: the model card's standard error is about 620 points. The golden
  trajectory is the real gate.
