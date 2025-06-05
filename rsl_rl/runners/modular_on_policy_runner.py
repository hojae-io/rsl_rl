#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import pickle
from collections import defaultdict
from pathlib import Path

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import store_code_state


class ModularOnPolicyRunner:
    """Modular On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.leg_alg_cfg = train_cfg["leg_algorithm"]
        self.leg_policy_cfg = train_cfg["leg_policy"]
        self.arm_alg_cfg = train_cfg["arm_algorithm"]
        self.arm_policy_cfg = train_cfg["arm_policy"]
        self.device = device
        self.env = env

        print("\n--------------- Create leg actor critic ---------------")
        leg_actor_critic_class = eval(self.leg_policy_cfg.pop("class_name"))  # ActorCritic
        leg_actor_critic: ActorCritic = leg_actor_critic_class(
            self.env.num_actor_obs["leg"], self.env.num_critic_obs["leg"], self.env.num_actions["leg"], **self.leg_policy_cfg
        ).to(self.device)
        leg_alg_class = eval(self.leg_alg_cfg.pop("class_name"))  # PPO
        self.leg_alg: PPO = leg_alg_class(leg_actor_critic, device=self.device, **self.leg_alg_cfg)

        print("\n--------------- Create arm actor critic ---------------")
        arm_actor_critic_class = eval(self.arm_policy_cfg.pop("class_name"))  # ActorCritic
        arm_actor_critic: ActorCritic = arm_actor_critic_class(
            self.env.num_actor_obs["arm"], self.env.num_critic_obs["arm"], self.env.num_actions["arm"], **self.arm_policy_cfg
        ).to(self.device)
        arm_alg_class = eval(self.arm_alg_cfg.pop("class_name"))  # PPO
        self.arm_alg: PPO = arm_alg_class(arm_actor_critic, device=self.device, **self.arm_alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # * init storage and model
        self.leg_alg.init_storage(self.env.num_envs,
                                  self.num_steps_per_env,
                                  self.env.num_actor_obs["leg"],
                                  self.env.num_critic_obs["leg"],
                                  self.env.num_actions["leg"])
        
        self.arm_alg.init_storage(self.env.num_envs,
                                  self.num_steps_per_env,
                                  self.env.num_actor_obs["arm"],
                                  self.env.num_critic_obs["arm"],
                                  self.env.num_actions["arm"])

        # * Log
        self.log_dir = log_dir
        self.log_buffer: dict[str, list] = defaultdict(list)
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # * initialize writer
        if self.log_dir is not None and self.writer is None and self.cfg["enable_logging"]:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, 
                                       {'leg_alg_cfg': self.leg_alg_cfg, 'arm_alg_cfg': self.arm_alg_cfg},
                                       {'leg_policy_cfg': self.leg_policy_cfg, 'arm_policy_cfg': self.arm_policy_cfg})
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, 
                                       {'leg_alg_cfg': self.leg_alg_cfg, 'arm_alg_cfg': self.arm_alg_cfg},
                                       {'leg_policy_cfg': self.leg_policy_cfg, 'arm_policy_cfg': self.arm_policy_cfg})

            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_dict = self.env.get_observations()
        leg_actor_obs, leg_critic_obs = obs_dict["leg_actor"], obs_dict["leg_critic"]
        arm_actor_obs, arm_critic_obs = obs_dict["arm_actor"], obs_dict["arm_critic"]
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = {"leg": deque(maxlen=100), "arm": deque(maxlen=100)}
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = {"leg": torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device),
                          "arm": torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)}
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        if self.cfg["enable_logging"]:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter+1, tot_iter+1):
            start = time.time()
            # * Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    leg_actions = self.leg_alg.act(leg_actor_obs, leg_critic_obs)
                    arm_actions = self.arm_alg.act(arm_actor_obs, arm_critic_obs)
                    actions = torch.cat((leg_actions, arm_actions), dim=1)

                    obs_dict, rewards, dones, terminated, time_outs, infos = self.env.step(actions)

                    leg_actor_obs, leg_critic_obs = obs_dict["leg_actor"], obs_dict["leg_critic"]
                    arm_actor_obs, arm_critic_obs = obs_dict["arm_actor"], obs_dict["arm_critic"]

                    # self.leg_alg.process_env_step(rewards, dones, time_outs | terminated["arm"])
                    # self.arm_alg.process_env_step(rewards, dones, time_outs | terminated["leg"]) # TODO: This seems worse

                    # self.leg_alg.process_env_step(rewards, dones, time_outs)
                    # self.arm_alg.process_env_step(rewards, dones, time_outs)

                    # leg_dones = dones.clone()
                    # leg_dones[terminated["arm"]] = 0.0
                    # arm_dones = dones.clone()
                    # arm_dones[terminated["leg"]] = 0.0
                    self.leg_alg.process_env_step(rewards["leg"], dones, time_outs)
                    self.arm_alg.process_env_step(rewards["arm"], dones, time_outs)

                    if self.log_dir is not None:
                        # * Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum["leg"] += rewards["leg"]
                        cur_reward_sum["arm"] += rewards["arm"]
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer["leg"].extend(cur_reward_sum["leg"][new_ids][:, 0].cpu().numpy().tolist())
                        rewbuffer["arm"].extend(cur_reward_sum["arm"][new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum["leg"][new_ids] = 0
                        cur_reward_sum["arm"][new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # * Learning step
                start = stop
                self.leg_alg.compute_returns(leg_critic_obs)
                self.arm_alg.compute_returns(arm_critic_obs)

            leg_mean_value_loss, leg_mean_surrogate_loss = self.leg_alg.update()
            arm_mean_value_loss, arm_mean_surrogate_loss = self.arm_alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if (it % self.save_interval == 0) and self.cfg["enable_logging"]:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == (start_iter+1) and self.cfg.get("store_code_state", True) and self.cfg["enable_logging"]:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
        if (it % self.save_interval != 0) and self.cfg["enable_logging"]:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 100, pad: int = 45):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    if self.cfg["enable_logging"]:
                        self.writer.add_scalar(key, value, locs["it"])
                        self.log_buffer[key].append(value.item())
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    if self.cfg["enable_logging"]:
                        self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.leg_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        if self.cfg["enable_logging"]:
            self.writer.add_scalar("Loss/leg/value_function", locs["leg_mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/leg/surrogate", locs["leg_mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/arm/value_function", locs["arm_mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/arm/surrogate", locs["arm_mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/learning_rate", self.leg_alg.learning_rate, locs["it"])
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
            self.writer.add_scalar("Policy/leg/advantage_variance", self.leg_alg.storage.raw_advantages.var(), locs["it"])
            self.writer.add_scalar("Policy/arm/advantage_variance", self.arm_alg.storage.raw_advantages.var(), locs["it"])
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            if len(locs["rewbuffer"]["leg"]) > 0:
                self.writer.add_scalar("Train/mean_reward/leg", statistics.mean(locs["rewbuffer"]["leg"]), locs["it"])
                self.writer.add_scalar("Train/mean_reward/arm", statistics.mean(locs["rewbuffer"]["arm"]), locs["it"])
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
                if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                    self.writer.add_scalar("Train/mean_reward/leg/time", statistics.mean(locs["rewbuffer"]["leg"]), self.tot_time)
                    self.writer.add_scalar("Train/mean_reward/arm/time", statistics.mean(locs["rewbuffer"]["arm"]), self.tot_time)
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                    )

            self.log_buffer["Policy/leg/advantage_variance"].append(self.leg_alg.storage.raw_advantages.var().item())
            self.log_buffer["Policy/arm/advantage_variance"].append(self.arm_alg.storage.raw_advantages.var().item())
            self.log_buffer["Train/mean_reward/leg"].append(statistics.mean(locs["rewbuffer"]["leg"]))
            self.log_buffer["Train/mean_reward/arm"].append(statistics.mean(locs["rewbuffer"]["arm"]))

        # Video recording for wandb
        if self.cfg["enable_logging"] and self.logger_type == "wandb":
            self.writer.update_video_files(log_name="Video", fps=30)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]["leg"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss/leg:':>{pad}} {locs['leg_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/leg:':>{pad}} {locs['leg_mean_surrogate_loss']:.4f}\n"""
                f"""{'Value function loss/arm:':>{pad}} {locs['arm_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/arm:':>{pad}} {locs['arm_mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/leg:':>{pad}} {statistics.mean(locs['rewbuffer']["leg"]):.2f}\n"""
                f"""{'Mean reward/arm:':>{pad}} {statistics.mean(locs['rewbuffer']["arm"]):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss/leg:':>{pad}} {locs['leg_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/leg:':>{pad}} {locs['leg_mean_surrogate_loss']:.4f}\n"""
                f"""{'Value function loss/arm:':>{pad}} {locs['arm_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/arm:':>{pad}} {locs['arm_mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "leg_model_state_dict": self.leg_alg.actor_critic.state_dict(),
            "leg_optimizer_state_dict": self.leg_alg.optimizer.state_dict(),
            "arm_model_state_dict": self.arm_alg.actor_critic.state_dict(),
            "arm_optimizer_state_dict": self.arm_alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }

        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

        exp_name = self.cfg["experiment_name"]
        seed     = self.cfg["seed"]
        filename = f"{exp_name}_seed_{seed}_log_buffer.pkl"
        log_dir_path = Path(self.log_dir) / filename
        data_dir_path = Path.cwd() / "scripts" / "plotting" / "data" / exp_name / filename

        # 1. Primary location inside self.log_dir
        with log_dir_path.open("wb") as f:
            pickle.dump(dict(self.log_buffer), f)

        # 2. Mirror copy inside scripts/plotting/data/<experiment_name>/
        with data_dir_path.open("wb") as f:
            pickle.dump(dict(self.log_buffer), f)

    def load(self, path, load_modular: bool = True, load_optimizer: bool = True, only_leg: bool = False):
        try:
            loaded_dict = torch.load(path)
        except:
            import sys
            sys.modules['learning'] = sys.modules['rsl_rl']
            sys.modules['learning.storage'] = sys.modules['rsl_rl.storage']
            loaded_dict = torch.load(path)

        if load_modular:
            if only_leg:
                self.leg_alg.actor_critic.load_state_dict(loaded_dict["leg_model_state_dict"])
                if load_optimizer:
                    self.leg_alg.optimizer.load_state_dict(loaded_dict["leg_optimizer_state_dict"])
                self.current_learning_iteration = loaded_dict["iter"]
            else:
                self.leg_alg.actor_critic.load_state_dict(loaded_dict["leg_model_state_dict"])
                self.arm_alg.actor_critic.load_state_dict(loaded_dict["arm_model_state_dict"])
                if load_optimizer:
                    self.leg_alg.optimizer.load_state_dict(loaded_dict["leg_optimizer_state_dict"])
                    self.arm_alg.optimizer.load_state_dict(loaded_dict["arm_optimizer_state_dict"])
                self.current_learning_iteration = loaded_dict["iter"]
        else:
            self.leg_alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
            if load_optimizer:
                self.leg_alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.current_learning_iteration = loaded_dict["iter"]

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.leg_alg.actor_critic.to(device)
            self.arm_alg.actor_critic.to(device)
        leg_policy = self.leg_alg.actor_critic.act_inference
        arm_policy = self.arm_alg.actor_critic.act_inference
        return leg_policy, arm_policy

    def train_mode(self):
        self.leg_alg.actor_critic.train()
        self.arm_alg.actor_critic.train()

    def eval_mode(self):
        self.leg_alg.actor_critic.eval()
        self.arm_alg.actor_critic.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def export(self, path, model_name):
        self.leg_alg.actor_critic.export_policy(path, model_name + "_leg")
        self.arm_alg.actor_critic.export_policy(path, model_name + "_arm")

    def close(self):
        if self.writer is not None:
            self.writer.stop()