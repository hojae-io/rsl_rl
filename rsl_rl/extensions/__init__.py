# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extensions for the learning algorithms."""

from .amp import AdversarialMotionPrior, resolve_amp_config
from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .symmetry import Symmetry, resolve_symmetry_config

__all__ = [
    "AdversarialMotionPrior",
    "RandomNetworkDistillation",
    "Symmetry",
    "resolve_amp_config",
    "resolve_rnd_config",
    "resolve_symmetry_config",
]
