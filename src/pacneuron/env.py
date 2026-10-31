"""Atari Ms. Pac-Man environment exactly as the pretrained Sample Factory agents saw it during training.

Local: hand-ported wrappers + `make_env()` + a deterministic `replay()`.
Global: rollout.py runs many of these envs in worker processes; features.py consumes the per-step
ground-truth record (RAM, raw reward, lives) that `GroundTruthInfo` attaches to `info`.

Wrapper chain (Sample Factory sf_examples/atari/atari_utils.py::make_atari_env, MIT):
  gym.make("MsPacmanNoFrameskip-v4") -> RecordEpisodeStatistics -> NoopResetEnv(30) -> MaxAndSkipEnv(4)
  -> EpisodicLifeEnv -> ClipRewardEnv -> ResizeObservation(84,84) -> GrayScaleObservation -> FrameStack(4)
NoopResetEnv / MaxAndSkipEnv / EpisodicLifeEnv / ClipRewardEnv are ported verbatim in behaviour from
sample_factory/envs/env_wrappers.py (themselves from CleanRL / Stable-Baselines3).
Resize/Grayscale reproduce gym 0.26 semantics with OpenCV: cv2.resize(..., INTER_AREA) then
cv2.cvtColor(RGB2GRAY) (https://github.com/openai/gym/blob/0.26.2/gym/wrappers/gray_scale_observation.py);
gymnasium>=1.0's GrayscaleObservation instead computes np.sum(obs * [0.2125, 0.7154, 0.0721]).astype(uint8)
(different weights, truncation; gymnasium/wrappers/transform_observation.py), which would shift input pixels, so we
do not use it.
"""

from __future__ import annotations

from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np

from pacneuron.config import ENV_ID, STICKY

gym.register_envs(ale_py)  # gymnasium>=1.0 does not auto-register ALE ids (https://ale.farama.org/gymnasium-interface/)


class NoopResetEnv(gym.Wrapper):
    """On reset take 1..noop_max NOOPs (count drawn from the env's seeded np_random)."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))  # same RNG call as SF/SB3
        obs, info = np.zeros(0), {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated | truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat the action `skip` frames, sum rewards, return the pixel-wise max of the last two frames."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        total_reward, info = 0.0, {}
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated | truncated:
                break
        # As in SF/SB3: on an early break the buffer may hold stale frames; that obs is never used for acting.
        return self._obs_buffer.max(axis=0), total_reward, terminated, truncated, info


class GroundTruthInfo(gym.Wrapper):
    """Attach emulator ground truth to `info` after every agent step (not part of the agent's input).

    Placed right after MaxAndSkipEnv so `raw_reward` is the unclipped 4-frame sum and `ram` is the
    console RAM after the last of the 4 frames. `rgb` is the max-pooled RGB frame the agent's
    observation is computed from (used for colour-based labels like frightened ghosts).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, raw_reward=float(reward), ram=self.unwrapped.ale.getRAM().copy(),
                    lives=int(self.unwrapped.ale.lives()), rgb=obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info, raw_reward=0.0, ram=self.unwrapped.ale.getRAM().copy(),
                    lives=int(self.unwrapped.ale.lives()), rgb=obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Life loss => terminated=True; real reset only on game over, otherwise one NOOP step."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated | truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)  # advance past the life-loss frame
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """Reward -> sign(reward) (training-time reward; raw reward is kept in info by GroundTruthInfo)."""

    def reward(self, reward):
        return np.sign(reward)


class ResizeGrayStack(gym.Wrapper):
    """gym-0.26 ResizeObservation(84,84) -> GrayScaleObservation -> FrameStack(4), returning uint8 [4,84,84].

    FrameStack semantics (gym 0.26 / gymnasium 0.2x): on reset the stack is filled with copies of the
    reset frame; on step the newest frame is appended.
    """

    def __init__(self, env: gym.Env, size: int = 84, k: int = 4):
        super().__init__(env)
        self.size, self.k = size, k
        self.frames: deque = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(0, 255, (k, size, size), np.uint8)

    def _process(self, rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)  # gym ResizeObservation
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)  # gym 0.26 GrayScaleObservation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        f = self._process(obs)
        for _ in range(self.k):
            self.frames.append(f)
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._process(obs))
        return np.stack(self.frames), reward, terminated, truncated, info


def _find_wrapper(env: gym.Env, cls: type) -> gym.Wrapper:
    """Walk the wrapper chain outward-in and return the first wrapper of type `cls`."""
    while not isinstance(env, cls):
        env = env.env
    return env


def make_env(sticky: float = 0.0, render_mode: str | None = None) -> gym.Env:
    """Build the training-identical env. `sticky` > 0 enables ALE sticky actions (prereg fallback only)."""
    env = gym.make(ENV_ID, repeat_action_probability=sticky, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # innermost, as in SF: records true game score/length
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = GroundTruthInfo(env)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    return ResizeGrayStack(env)


class Game:
    """One full game (all lives) driven step by step, mirroring how Sample Factory's sampler resets.

    SF resets an env whenever `terminated or truncated` is returned; with EpisodicLifeEnv that happens on
    every life loss (a NOOP-advance, not a real reset). `step()` hides this: it returns the next obs to act
    on, the info of the step taken, whether a life was lost, and whether the real game is over.
    """

    def __init__(self, seed: int, sticky: float = STICKY):
        self.env = make_env(sticky)
        self._life = _find_wrapper(self.env, EpisodicLifeEnv)  # knows whether a termination was real
        # obs = stacked frames the agent acts on next; info = ground truth (RAM, rgb, ...) of that same state.
        self.obs, self.info = self.env.reset(seed=seed)
        self.over = False

    def step(self, action: int):
        """Act; returns (info of the transition, life_lost, game_over). self.obs/self.info move to the next state."""
        obs, _, terminated, truncated, info = self.env.step(int(action))
        game_over = bool(terminated or truncated) and self._life.was_real_done
        life_lost = bool(terminated or truncated) and not game_over
        next_info = info
        if life_lost:
            obs, next_info = self.env.reset()  # EpisodicLifeEnv: NOOP-advance and refill the frame stack
        self.obs, self.info, self.over = obs, next_info, game_over
        return info, life_lost, game_over

    def close(self):
        self.env.close()


def replay(seed: int, actions: np.ndarray, capture: set[int] | None = None, sticky: float = STICKY) -> dict:
    """Deterministically re-run a recorded game; return RGB frames/obs/RAM at the requested step indices.

    Step index t refers to the state the agent acted on at its t-th decision (t=0 after reset).
    Used to re-render top-activating frames instead of storing all frames on disk.
    """
    game, out = Game(seed, sticky), {}
    for t, a in enumerate(actions):
        if capture is None or t in capture:
            out[t] = {"obs": game.obs.copy(), "rgb": game.info["rgb"].copy(), "ram": game.info["ram"].copy()}
        if game.over:
            break
        game.step(a)
    game.close()
    return out
