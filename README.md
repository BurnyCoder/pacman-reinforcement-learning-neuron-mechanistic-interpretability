# Ms. Pac-Man neuron microscope: neuron-level interpretability and steering of reward-trained RL agents

**What:** this repo applies neuron-level mechanistic interpretability, in the style of Goodfire's
["Neural Networks Are Already Clean"](https://already-clean.goodfire.com/), to deep reinforcement-learning agents
that learned Atari **Ms. Pac-Man purely from game reward** (no imitation learning, no human data). It checks
whether individual neurons have one readable job, and whether steering single neurons or channels changes
behaviour in predictable ways.

**How:**
- Load pretrained [Sample Factory](https://github.com/alex-petrenko/sample-factory) APPO agents:
  [seed 1111](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_1111), 2222, 3333, each trained for
  2 billion env steps.
- Re-implement them in plain PyTorch, verified bit-exact against real Sample Factory.
- Record every neuron while they play.
- Label each neuron against **exact game state decoded from the Atari console RAM**: ghost distances and
  directions in maze steps, frightened mode, pellets, open directions.
- Test the labels causally by steering neurons during closed-loop play.

**Status:** under construction. The pipeline, fidelity gate and preregistration are done; results are not in
yet. Hypotheses, metrics and falsification criteria are fixed in advance in
[`reports/preregistration.md`](reports/preregistration.md).

## Motivation
- Prior interpretability work on Ms. Pac-Man agents:
  - pixel saliency ([Greydanus et al. 2018](https://arxiv.org/abs/1711.00138));
  - attention maps ([Mott et al. 2019](https://arxiv.org/abs/1906.02500));
  - NMF concepts ([Bekkemoen & Langseth 2025](https://mlanthology.org/mlj/2025/bekkemoen2025mlj-interpretable));
  - a sparse autoencoder without interventions
    ([DuPlessie 2024](https://math.mit.edu/research/highschool/primes/materials/2024/DuPlessie.pdf)).
- Causal steering of single channels in RL agents has been shown for:
  - the Procgen maze agent ([Mini, Turner et al. 2023](https://arxiv.org/abs/2310.08043): "cheese channel"
    retargeting);
  - Sokoban planners ([Bush et al. 2025](https://arxiv.org/abs/2504.01871);
    [Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421)).
- We found no published neuron-level steering of a reward-trained Ms. Pac-Man agent.

## Method (summary)
1. **Fidelity gate.**
   - `scripts/sf_golden_wsl.sh` builds the training-era stack in WSL (sample-factory 2.0.3, gym 0.26.2,
     ale-py 0.8.1) and records a golden trajectory.
   - `tests/test_fidelity.py` checks that our Windows environment wrapper chain gives bit-identical
     observations, and that our model gives logits within 1e-5.
2. **Ground truth.**
   - RAM addresses come from [OCAtari](https://github.com/k4ntz/OC_Atari) and
     [AtariARI](https://github.com/mila-iqia/atari-representation-learning).
   - The frightened-mode state was found here: `ram[116]` is a countdown timer, and bit 7 of `ram[1+i]` marks
     ghost i as edible. Both are validated against the blue sprite pixels.
   - Maze (BFS) distances run on a 14x18 lattice graph extracted from the frame.
3. **Neuron audit.** For all 512 fc neurons and the 64 conv3 channels:
   - positive and negative poles and top-activating frames;
   - single-feature R² against a gradient-boosting ceiling, giving a "one main job" score;
   - direct logit attribution;
   - controls: random directions, random non-negative unit mixtures, an untrained network.
4. **Steering.**
   - Clamp, ablate or spatially patch units during play, with common random numbers (paired seeds).
   - Compare against matched random-unit nulls on held-out games, using the preregistered metrics.

## Setup and usage
```bash
uv sync                                   # Python 3.12 venv in .venv (torch CUDA 13.0 wheels)
cp .env.example .env                      # set DATA_DIR / HF_HOME (bulk data lives outside the repo)
uv run pytest                             # fidelity gate + unit tests
uv run python -m pacneuron.pipeline --phase all --seed 1111
```
Optional golden-fixture rebuild (Linux or WSL): `bash scripts/sf_golden_wsl.sh /mnt/c/<path-to-repo>`.

## Repository layout
- `src/pacneuron/`:
  - `env.py`: the training-identical wrapper chain;
  - `model.py`: the plain-PyTorch agent and interventions;
  - `ram.py`, `maze.py`, `features.py`: ground truth;
  - `rollout.py`: parallel play with batched GPU inference.
- `tests/`: the fidelity gate and unit tests.
- `scripts/`: the WSL golden-fixture builder.
- `reports/`: the preregistration and per-experiment reports.

## Technologies
- PyTorch 2.14 (CUDA 13.0), Gymnasium 1.4, ale-py 0.12.
- NumPy, SciPy, scikit-learn, Hugging Face Hub.
- uv, pytest.
