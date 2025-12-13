#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import rsl_rl
from rsl_rl.env import VecEnv


class PolicyRunner(ABC):
    """Abstract base class for policy runners.

    This class defines the common interface and shared functionality for all policy runners,
    including OnPolicyRunner, ModularOnPolicyRunner, and future OffPolicyRunner implementations.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        """Initialize the policy runner.

        Args:
            env: Vectorized environment.
            train_cfg: Training configuration dictionary.
            log_dir: Directory for logging. Defaults to None.
            device: Device to run on. Defaults to "cpu".
        """
        self.cfg = train_cfg
        self.device = device
        self.env = env
        self.num_steps_per_env = train_cfg.get("num_steps_per_env", 1)
        self.save_interval = train_cfg.get("save_interval", 100)

        # * Log
        self.log_dir = log_dir
        self.log_buffer: dict[str, list] = defaultdict(list)
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    @abstractmethod
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Train the policy for a specified number of learning iterations.

        Args:
            num_learning_iterations: Number of learning iterations to perform.
            init_at_random_ep_len: Whether to initialize at random episode length. Defaults to False.
        """
        pass

    @abstractmethod
    def log(self, locs: dict, width: int = 100, pad: int = 45) -> None:
        """Log training metrics and statistics.

        Args:
            locs: Dictionary of local variables from the learning loop.
            width: Width of the log output. Defaults to 100.
            pad: Padding for log output. Defaults to 45.
        """
        pass

    @abstractmethod
    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the model checkpoint.

        Args:
            path: Path to save the checkpoint.
            infos: Additional information to save. Defaults to None.
        """
        pass

    @abstractmethod
    def load(self, path: str, **kwargs) -> dict | None:
        """Load a model checkpoint.

        Args:
            path: Path to the checkpoint file.
            **kwargs: Additional loading arguments (e.g., load_optimizer, load_modular, only_leg).

        Returns:
            Dictionary of additional information from the checkpoint, or None.
        """
        pass

    @abstractmethod
    def get_inference_policy(self, device: str | None = None):
        """Get the inference policy for evaluation.

        Args:
            device: Device to run inference on. Defaults to None (uses self.device).

        Returns:
            Inference policy function(s). Return type depends on the specific runner implementation.
        """
        pass

    @abstractmethod
    def train_mode(self) -> None:
        """Switch to training mode (e.g., enables dropout)."""
        pass

    @abstractmethod
    def eval_mode(self) -> None:
        """Switch to evaluation mode (e.g., disables dropout)."""
        pass

    @abstractmethod
    def export(self, path: str, model_name: str) -> None:
        """Export the policy model for deployment.

        Args:
            path: Directory path to export the model.
            model_name: Name for the exported model.
        """
        pass

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Add a git repository to track for code state logging.

        Args:
            repo_file_path: Path to the repository file to track.
        """
        self.git_status_repos.append(repo_file_path)

    def close(self) -> None:
        """Close the runner and clean up resources (e.g., close log writers)."""
        if self.writer is not None:
            self.writer.stop()
