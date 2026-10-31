# Experiment 1: Fidelity gate (does our code run the pretrained agent exactly as it was trained?)

## Why
Sample Factory 2.0.x has no native Windows support ("There is no Windows support at this time",
[installation docs](https://www.samplefactory.dev/01-get-started/installation/)). Without a check, a
re-implementation could silently differ from the agent that was actually trained, and every neuron label would then
describe a different network. Two common pitfalls:
- gymnasium ≥ 1.0 changed its grayscale wrapper from `cv2.cvtColor` to NumPy weights
  (`0.2125/0.7154/0.0721`, truncating; `gymnasium/wrappers/transform_observation.py`). This shifts input pixels by up
  to a few gray levels.
- The observation normaliser (running mean/std, float64 buffers cast to float32, clip ±5) must be applied exactly.

## Method
- **Reference.** `scripts/sf_golden_wsl.sh` builds the training-era stack in WSL: sample-factory 2.0.3, gym 0.26.2,
  ale-py 0.8.1, torch 1.13.1 (CPU). The pretrained models were uploaded on 2022-10-20, before sample-factory 2.0.0
  was released on PyPI (2022-12-01), so no released version is exactly the training code. 2.0.3 (2023-02-09) is the
  last release built on gym 0.26 (2.1 moved to gymnasium; PyPI metadata). The Atari wrapper file
  `sf_examples/atari/atari_utils.py` is byte-identical in the v2.0.0 and v2.0.2 git tags; 2.0.3 has no git tag. With Sample Factory's own `make_atari_env` and `create_actor_critic`, the script plays a fixed
  random action sequence (env seed 123) and records observations, logits and values.
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
| max abs. logit difference (CPU) | 1.1e-05 |
| max abs. value difference (CPU) | 4.8e-07 |
| max abs. logit difference on GPU (NVIDIA GeForce RTX 5070 Laptop GPU, cuDNN TF32 on, as in data collection) | 4.5e-02 |
| reference stack | sample-factory 2.0.3, gym 0.26.2, ale-py 0.8.1, torch 1.13.1 (WSL) |
<!-- END:fidelity -->

The gate is enforced by `tests/test_fidelity.py`: observations must be bit-exact; logits and values must pass
`assert_allclose` with atol = rtol = 1e-5. The CPU logit difference of 1.1e-5 is just above 1e-5 in absolute terms
and passes through the relative tolerance. The test runs on CPU only. On the GPU used for every rollout, PyTorch runs
cuDNN convolutions in TF32 by default (`torch.backends.cudnn.allow_tf32 = True`,
[PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)),
and the logits then differ by up to <!--n:fid_logit_gpu-->4.5e-02<!--/n--> (table above). The unsteered A/A arm
(`aa-control`) reproduces <!--n:appo_1111:aa_identical-->50 of 50<!--/n--> paired baseline games action for action,
so this does not break the paired design. Still, the analysed agent is a close numerical neighbour of the trained
one, not a bit-exact copy.

## Learnings
- ale-py 0.12 still registers `MsPacmanNoFrameskip-v4` with `max_num_frames_per_episode=108000`, the same as ale-py
  0.8.1 ([`src/python/gym.py` at v0.8.1](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.8.1/src/python/gym.py)).
  The episode limit therefore matches training.
- Rewriting the Atari wrappers by hand (`src/pacneuron/env.py`) proved safer than relying on any library version's
  wrapper semantics.
- A score check alone would be a weak gate: the model card reports 6288 ± 1955 (SD) without the number of episodes,
  and our own 100-game scores vary by thousands of points. The golden trajectory is the real gate.
- The gate must cover the device used for inference. A CPU-only gate missed the TF32 difference above.
