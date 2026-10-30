"""Generate golden fixtures with the REAL Sample Factory stack (run inside WSL/Linux, not Windows).

Local: builds the env with Sample Factory's own `make_atari_env` (gym 0.26 + ale-py 0.8.1) and the model
with SF's own `create_actor_critic`, plays a fixed action sequence, and saves observations, logits and
values to tests/fixtures/sf_golden.npz.
Global: tests/test_fidelity.py checks that our Windows re-implementation (env.py, model.py) reproduces
these observations bit-exactly and these logits to 1e-5 - the fidelity gate for every later result.

Setup (see scripts/sf_golden_wsl.sh): Python 3.10 venv with sample-factory==2.0.3, gym[atari,accept-rom-license]==0.26.2,
torch<2 (CPU). The pretrained agents were trained in Oct 2022 with the SF 2.0 pre-release (gym 0.26 era).
Usage: python scripts/sf_golden_wsl.py <checkpoint.pth> <cfg.json> <out.npz>
"""

import json
import sys
from types import SimpleNamespace

import gym
import numpy as np
import torch
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sf_examples.atari.atari_utils import make_atari_env
from sf_examples.atari.train_atari import parse_atari_args

ckpt_path, cfg_path, out_path = sys.argv[1:4]
SEED, N_STEPS = 123, 600  # fixed env seed and number of agent steps (random actions -> several life losses)

# Full SF config: defaults from the atari arg parser, then every value saved in the run's cfg.json.
cfg = parse_atari_args(argv=["--env=atari_mspacman", "--experiment=golden"], evaluation=True)
for k, v in json.load(open(cfg_path)).items():
    setattr(cfg, k, v)

env = make_atari_env("atari_mspacman", cfg, env_config=None)  # SF's exact wrapper chain
obs_space = gym.spaces.Dict({"obs": env.observation_space})
ac = create_actor_critic(cfg, obs_space, env.action_space)
ac.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
ac.eval()

rng = np.random.default_rng(0)
actions = rng.integers(0, env.action_space.n, size=N_STEPS)
obs, _ = env.reset(seed=SEED)
all_obs, life_lost, game_over = [], [], []
for t in range(N_STEPS):
    all_obs.append(np.asarray(obs, dtype=np.uint8))
    obs, rew, terminated, truncated, info = env.step(int(actions[t]))
    done = terminated or truncated
    real = env.unwrapped.ale.game_over()
    life_lost.append(bool(done and not real))
    game_over.append(bool(done and real))
    if done and real:
        break
    if done:
        obs, _ = env.reset()  # SF resets on every termination; EpisodicLifeEnv turns this into a NOOP-advance

obs_arr = np.stack(all_obs)
with torch.no_grad():
    x = {"obs": torch.from_numpy(obs_arr).float()}  # float copy: normalizer scales in place
    normalized = ac.normalize_obs(x)
    rnn = torch.zeros(len(obs_arr), get_rnn_size(cfg))
    res = ac(normalized, rnn)
np.savez_compressed(out_path, obs=obs_arr, actions=actions[: len(obs_arr)], seed=SEED,
                    logits=res["action_logits"].numpy(), values=res["values"].numpy(),
                    life_lost=np.array(life_lost), game_over=np.array(game_over))
print("saved", out_path, obs_arr.shape, "life losses", int(np.sum(life_lost)), "game over", any(game_over))
