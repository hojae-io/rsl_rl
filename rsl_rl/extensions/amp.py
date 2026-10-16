# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, NoReturn

from rsl_rl.env import VecEnv
from rsl_rl.modules import MLP, EmpiricalNormalization


class AdversarialMotionPrior(nn.Module):
    """Implementation of Adversarial Motion Priors (AMP).

    A discriminator is trained to separate transitions of a reference motion dataset from transitions of the policy.
    Its output on the policy transitions is turned into a style reward that is added to the task reward.

    Reference:
        - Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control." SIGGRAPH (2021).
    """

    def __init__(
        self,
        reference: torch.Tensor,
        obs_groups: dict,
        num_envs: int,
        num_steps: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512),
        activation: str = "relu",
        learning_rate: float = 1e-4,
        weight: float = 1.0,
        gradient_penalty: float = 5.0,
        device: str = "cpu",
    ) -> None:
        """Initialize the AMP module.

        Args:
            reference: Reference transitions, each the AMP states of two consecutive frames concatenated. Shape is
                (num_transitions, 2 * num_states). Its statistics fix the input normalization of the discriminator.
            obs_groups: Dictionary of observation groups.
            num_envs: Number of environments.
            num_steps: Number of environment steps per learning iteration.
            hidden_dims: List of hidden dimensions of the discriminator.
            activation: Activation function.
            learning_rate: Learning rate for the discriminator optimizer.
            weight: Scaling factor of the style reward.
            gradient_penalty: Coefficient of the gradient penalty on the reference transitions.
            device: Device to use.
        """
        super().__init__()
        self.obs_groups = obs_groups
        self.weight = weight
        self.gradient_penalty = gradient_penalty
        self.device = device
        self.reference = reference.to(self.device)
        num_inputs = self.reference.shape[1]

        self.normalizer = EmpiricalNormalization(shape=[num_inputs]).to(self.device)
        self.normalizer.update(self.reference)
        self.discriminator = MLP(num_inputs, 1, hidden_dims, activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

        # Policy transitions of the current rollout and whether they stay within one episode
        self.transitions = torch.zeros(num_steps, num_envs, num_inputs, device=self.device)
        self.valid = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=self.device)
        self.step = 0

    def get_style_reward(self, obs: TensorDict, next_obs: TensorDict, dones: torch.Tensor) -> torch.Tensor:
        """Record the policy transition and compute its weighted style reward, zero where the episode ended."""
        transition = torch.cat([self.get_amp_state(obs), self.get_amp_state(next_obs)], dim=-1)
        valid = dones == 0
        self.transitions[self.step].copy_(transition)
        self.valid[self.step].copy_(valid)
        self.step += 1
        logits = self.discriminator(self.normalizer(transition)).squeeze(-1)
        return self.weight * torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0) * valid

    def compute_loss(self, batch_size: int) -> torch.Tensor:
        """Least-squares discriminator loss on sampled policy and reference transitions, with a gradient penalty."""
        valid = self.valid.flatten().nonzero().squeeze(-1)
        policy = self.transitions.flatten(0, 1)[valid[torch.randint(len(valid), (batch_size,), device=self.device)]]
        reference = self.reference[torch.randint(len(self.reference), (batch_size,), device=self.device)]
        reference = self.normalizer(reference).requires_grad_(True)
        policy_logits = self.discriminator(self.normalizer(policy))
        reference_logits = self.discriminator(reference)
        grad = torch.autograd.grad(reference_logits.sum(), reference, create_graph=True)[0]
        return (
            0.5 * torch.square(policy_logits + 1.0).mean()
            + 0.5 * torch.square(reference_logits - 1.0).mean()
            + self.gradient_penalty * grad.square().sum(dim=-1).mean()
        )

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> NoReturn:
        """Disallow generic forward calls for this module."""
        raise RuntimeError("Forward method is not implemented. Use get_style_reward instead.")

    def get_amp_state(self, obs: TensorDict) -> torch.Tensor:
        """Extract and concatenate observation groups used as the AMP state."""
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["amp"]], dim=-1)


def resolve_amp_config(
    alg_cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]], env: VecEnv, num_steps: int
) -> dict:
    """Resolve the AMP configuration.

    The environment provides the reference transitions as ``env.unwrapped.amp_reference``.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs: Observation dictionary.
        obs_groups: Observation groups dictionary.
        env: Environment object.
        num_steps: Number of environment steps per learning iteration.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    if alg_cfg.get("amp_cfg") is not None:
        num_states = sum(obs[obs_group].shape[-1] for obs_group in obs_groups["amp"])
        reference = env.unwrapped.amp_reference  # type: ignore
        if reference.shape[1] != 2 * num_states:
            raise ValueError(
                f"The AMP reference transitions have {reference.shape[1]} features but two AMP states have"
                f" {2 * num_states}."
            )
        alg_cfg["amp_cfg"]["reference"] = reference
        alg_cfg["amp_cfg"]["obs_groups"] = obs_groups
        alg_cfg["amp_cfg"]["num_envs"] = env.num_envs
        alg_cfg["amp_cfg"]["num_steps"] = num_steps
        # Scale down the weight with the timestep like the environment rewards
        alg_cfg["amp_cfg"]["weight"] *= env.unwrapped.step_dt  # type: ignore
    else:
        alg_cfg["amp_cfg"] = None
    return alg_cfg
