"""Pure-PyTorch re-implementation of the Sample Factory 2.0 `convnet_atari` actor-critic.

Local: `SFActorCritic` reproduces the exact forward pass of the pretrained APPO agents and exposes
every activation site (conv1-3, fc pre/post) plus an `edit` callback for interventions.
Global: Sample Factory itself cannot run on native Windows, so this module + `load_sf_checkpoint`
(strict key/shape mapping) is how the rest of the project runs the agent; fidelity is verified
against golden logits produced by real Sample Factory (tests/test_model.py).

Sources (Sample Factory, MIT):
- conv stack: sample_factory/model/encoder.py, convnet_atari = [[C,32,8,4],[32,64,4,2],[64,64,3,1]],
  ReLU after every conv, then flatten and `encoder_conv_mlp_layers=[512]` Linear+ReLU.
- obs normalisation: sample_factory/utils/normalize.py (obs * 1/255) then
  sample_factory/algo/utils/running_mean_std.py: x.sub_(mu).mul_(1/sqrt(var+1e-5)).clamp_(-5, 5),
  with mu/var cast to float32 (`.float()`).
- heads: critic_linear Linear(512,1); action_parameterization.distribution_linear Linear(512,9);
  no RNN (use_rnn=False), empty decoder (decoder_mlp_layers=[]), shared weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

# Running-mean-std epsilon and clip range used by Sample Factory (running_mean_std.py: _NORM_EPS, _DEFAULT_CLIP).
NORM_EPS = 1e-5
NORM_CLIP = 5.0

# An edit receives (site name, activation tensor) and returns the (possibly modified) activation.
EditFn = Callable[[str, torch.Tensor], torch.Tensor]

# Mapping from Sample Factory state_dict keys to our parameter names (strict: every key must be used).
_SF_KEYMAP = {
    "encoder.encoders.obs.enc.conv_head.0.weight": "conv1.weight",
    "encoder.encoders.obs.enc.conv_head.0.bias": "conv1.bias",
    "encoder.encoders.obs.enc.conv_head.2.weight": "conv2.weight",
    "encoder.encoders.obs.enc.conv_head.2.bias": "conv2.bias",
    "encoder.encoders.obs.enc.conv_head.4.weight": "conv3.weight",
    "encoder.encoders.obs.enc.conv_head.4.bias": "conv3.bias",
    "encoder.encoders.obs.enc.mlp_layers.0.weight": "fc.weight",
    "encoder.encoders.obs.enc.mlp_layers.0.bias": "fc.bias",
    "critic_linear.weight": "critic.weight",
    "critic_linear.bias": "critic.bias",
    "action_parameterization.distribution_linear.weight": "policy.weight",
    "action_parameterization.distribution_linear.bias": "policy.bias",
    "obs_normalizer.running_mean_std.running_mean_std.obs.running_mean": "obs_mean",
    "obs_normalizer.running_mean_std.running_mean_std.obs.running_var": "obs_var",
    "returns_normalizer.running_mean": "ret_mean",
    "returns_normalizer.running_var": "ret_var",
}
# Counters that carry no information for inference (number of samples seen by the normalisers).
_SF_IGNORED = {"obs_normalizer.running_mean_std.running_mean_std.obs.count", "returns_normalizer.count"}


class SFActorCritic(nn.Module):
    """Nature-CNN actor-critic with frozen observation normalisation and named activation sites.

    Sites (all post-ReLU unless noted), in forward order:
      conv1 [B,32,20,20], conv2 [B,64,9,9], conv3 [B,64,7,7], fc_pre [B,512] (pre-ReLU), fc [B,512].
    """

    SITES = ("conv1", "conv2", "conv3", "fc_pre", "fc")

    def __init__(self, n_actions: int = 9, frames: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(frames, 32, 8, stride=4)  # 84 -> 20
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)  # 20 -> 9
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)  # 9 -> 7
        self.fc = nn.Linear(64 * 7 * 7, 512)  # flatten in (C,H,W) order, as torch .view does in SF
        self.policy = nn.Linear(512, n_actions)  # action logits (Categorical over 9 actions)
        self.critic = nn.Linear(512, 1)  # value in *normalised* return units (normalize_returns=True)
        # Frozen normaliser statistics (buffers: saved in state_dict, never trained here).
        self.register_buffer("obs_mean", torch.zeros(frames, 84, 84, dtype=torch.float64))
        self.register_buffer("obs_var", torch.ones(frames, 84, 84, dtype=torch.float64))
        self.register_buffer("ret_mean", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("ret_var", torch.ones(1, dtype=torch.float64))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """uint8 [B,4,84,84] frames -> normalised float32 input, exactly as Sample Factory does."""
        x = obs.float() * (1.0 / 255.0)  # obs_scale=255 (normalize.py multiplies by 1/scale)
        mu = self.obs_mean.float()  # running_mean_std.py casts the float64 buffers to float32
        sigma = torch.sqrt(self.obs_var.float() + NORM_EPS)
        return ((x - mu) * (1.0 / sigma)).clamp(-NORM_CLIP, NORM_CLIP)

    def forward(self, obs: torch.Tensor, edit: EditFn | None = None) -> dict[str, torch.Tensor]:
        """Run the agent; `edit(site, act)` may replace any site's activation (interventions)."""
        e = edit or (lambda _site, a: a)  # identity edit when no intervention is requested
        x = self.normalize(obs)
        c1 = e("conv1", torch.relu(self.conv1(x)))
        c2 = e("conv2", torch.relu(self.conv2(c1)))
        c3 = e("conv3", torch.relu(self.conv3(c2)))
        h_pre = e("fc_pre", self.fc(c3.flatten(1)))
        h = e("fc", torch.relu(h_pre))
        return {"conv1": c1, "conv2": c2, "conv3": c3, "fc_pre": h_pre, "fc": h,
                "logits": self.policy(h), "value": self.critic(h).squeeze(-1)}

    def denormalize_value(self, v: torch.Tensor) -> torch.Tensor:
        """Map normalised critic output back to (clipped-reward) return units: v*std + mean."""
        return v * torch.sqrt(self.ret_var.float() + NORM_EPS) + self.ret_mean.float()


def load_sf_checkpoint(path, device: str = "cpu") -> SFActorCritic:
    """Load a Sample Factory checkpoint (.pth with a 'model' state_dict) into SFActorCritic, strictly.

    Raises if any SF key is unmapped, any of our tensors is missing, or a shape differs.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # SF pickles a plain dict
    sf_state = ckpt["model"]
    unknown = set(sf_state) - set(_SF_KEYMAP) - _SF_IGNORED
    if unknown:
        raise KeyError(f"Unmapped Sample Factory keys: {sorted(unknown)}")
    n_actions = sf_state["action_parameterization.distribution_linear.weight"].shape[0]
    model = SFActorCritic(n_actions=n_actions)
    state = {_SF_KEYMAP[k]: v for k, v in sf_state.items() if k in _SF_KEYMAP}
    model.load_state_dict(state, strict=True)  # strict: every parameter/buffer must be provided
    return model.to(device).eval()


def orthogonal_init_like(model: SFActorCritic, seed: int) -> SFActorCritic:
    """Untrained control network: same architecture and the *trained* obs normaliser, fresh weights.

    Sample Factory cfg uses policy_initialization=orthogonal with policy_init_gain=1.0 and zero biases
    (sample_factory/model/model_utils.py / actor_critic.py initialize_weights).
    """
    torch.manual_seed(seed)
    fresh = SFActorCritic(n_actions=model.policy.out_features)
    for layer in (fresh.conv1, fresh.conv2, fresh.conv3, fresh.fc, fresh.policy, fresh.critic):
        nn.init.orthogonal_(layer.weight, gain=1.0)
        nn.init.zeros_(layer.bias)
    for name in ("obs_mean", "obs_var", "ret_mean", "ret_var"):
        getattr(fresh, name).copy_(getattr(model, name))
    return fresh.to(next(model.parameters()).device).eval()


# ----------------------------------------------------------------------------------------------
# Interventions. Each is a callable EditFn; `rows` optionally restricts it to some batch rows
# (used for state-gated steering, e.g. "only while ghosts are not frightened").
# ----------------------------------------------------------------------------------------------


@dataclass
class Intervention:
    """Set/shift units at one site.

    mode:
      "set"  -> act[..., units] = value          (clamp; value=0 is zero-ablation)
      "add"  -> act[..., units] += value         (additive steering)
    For conv sites `units` indexes channels and `cell=(i, j)` optionally restricts to one spatial cell.
    `rows` is a bool tensor [B] (or None = all rows) selecting which environments are edited this step.
    """

    site: str
    units: list[int]
    value: float | torch.Tensor
    mode: str = "set"
    cell: tuple[int, int] | None = None
    rows: torch.Tensor | None = field(default=None, repr=False)

    def __call__(self, site: str, act: torch.Tensor) -> torch.Tensor:
        if site != self.site:
            return act
        out = act.clone()  # never modify the tensor other sites/consumers may hold
        idx = (slice(None), self.units) if self.cell is None else (slice(None), self.units, self.cell[0], self.cell[1])
        target = out[idx]
        new = torch.as_tensor(self.value, dtype=out.dtype, device=out.device).expand_as(target)
        new = new if self.mode == "set" else target + new
        if self.rows is not None:  # keep original values on rows where the gate is off
            gate = self.rows.to(out.device).view(-1, *([1] * (target.dim() - 1)))
            new = torch.where(gate, new, target)
        out[idx] = new
        return out


def compose(*edits: EditFn) -> EditFn:
    """Chain several edits (applied in order at every site)."""
    def _edit(site: str, act: torch.Tensor) -> torch.Tensor:
        for ed in edits:
            act = ed(site, act)
        return act
    return _edit
