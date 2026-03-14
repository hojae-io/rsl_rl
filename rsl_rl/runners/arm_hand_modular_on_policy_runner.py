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

from .policy_runner import PolicyRunner


class ArmHandModularOnPolicyRunner(PolicyRunner):
    """Arm-Hand Modular On-policy runner for training and evaluation.
    
    This runner handles the case where the hand policy is conditioned on the arm policy's action.
    The arm policy acts first, and its actions are concatenated to the hand policy's observations
    before the hand policy acts.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.arm_alg_cfg = train_cfg["algorithms"]["arm"]
        self.arm_policy_cfg = train_cfg["policies"]["arm"]
        self.hand_alg_cfg = train_cfg["algorithms"]["hand"]
        self.hand_policy_cfg = train_cfg["policies"]["hand"]

        print("\n--------------- Create arm actor critic ---------------")
        arm_actor_critic_class = eval(self.arm_policy_cfg.pop("class_name"))  # ActorCritic
        arm_actor_critic: ActorCritic = arm_actor_critic_class(
            self.env.num_actor_obs["arm_actor"], 
            self.env.num_critic_obs["arm_critic"], 
            self.env.num_actions["palm_vel"], 
            **self.arm_policy_cfg
        ).to(self.device)
        arm_alg_class = eval(self.arm_alg_cfg.pop("class_name"))  # PPO
        self.arm_alg: PPO = arm_alg_class(arm_actor_critic, device=self.device, **self.arm_alg_cfg)

        print("\n--------------- Create hand actor critic ---------------")
        # Hand policy input size = original hand obs + arm action size
        arm_action_dim = self.env.num_actions["palm_vel"]
        hand_actor_obs_dim = self.env.num_actor_obs["hand_actor"]
        hand_critic_obs_dim = self.env.num_critic_obs["hand_critic"]
        
        # Store original dimensions for conditioning
        self.arm_action_dim = arm_action_dim
        self.original_hand_actor_obs_dim = hand_actor_obs_dim
        self.original_hand_critic_obs_dim = hand_critic_obs_dim
        
        # Hand policy receives: hand_obs + arm_action
        hand_actor_critic_class = eval(self.hand_policy_cfg.pop("class_name"))  # ActorCritic
        hand_actor_critic: ActorCritic = hand_actor_critic_class(
            hand_actor_obs_dim + arm_action_dim,  # hand obs + arm action
            hand_critic_obs_dim + arm_action_dim,  # hand critic obs + arm action
            self.env.num_actions["fingertip_vel"], 
            **self.hand_policy_cfg
        ).to(self.device)
        hand_alg_class = eval(self.hand_alg_cfg.pop("class_name"))  # PPO
        self.hand_alg: PPO = hand_alg_class(hand_actor_critic, device=self.device, **self.hand_alg_cfg)

        # * init storage and model
        self.arm_alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            self.env.num_actor_obs["arm_actor"],
            self.env.num_critic_obs["arm_critic"],
            self.env.num_actions["palm_vel"]
        )
        
        self.hand_alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            hand_actor_obs_dim + arm_action_dim,  # hand obs + arm action
            hand_critic_obs_dim + arm_action_dim,  # hand critic obs + arm action
            self.env.num_actions["fingertip_vel"]
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # * initialize writer
        if self.log_dir is not None and self.writer is None and self.cfg["enable_logging"]:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(
                    self.env.cfg, self.cfg, 
                    {'arm_alg_cfg': self.arm_alg_cfg, 'hand_alg_cfg': self.hand_alg_cfg},
                    {'arm_policy_cfg': self.arm_policy_cfg, 'hand_policy_cfg': self.hand_policy_cfg}
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(
                    self.env.cfg, self.cfg, 
                    {'arm_alg_cfg': self.arm_alg_cfg, 'hand_alg_cfg': self.hand_alg_cfg},
                    {'arm_policy_cfg': self.arm_policy_cfg, 'hand_policy_cfg': self.hand_policy_cfg}
                )

            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_dict = self.env.get_observations()
        arm_actor_obs, arm_critic_obs = obs_dict["arm_actor"], obs_dict["arm_critic"]
        hand_actor_obs, hand_critic_obs = obs_dict["hand_actor"], obs_dict["hand_critic"]
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = {"arm": deque(maxlen=100), "hand": deque(maxlen=100)}
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = {
            "arm": torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device),
            "hand": torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        }
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
                    # Step 1: Get arm actions
                    arm_actions = self.arm_alg.act(arm_actor_obs, arm_critic_obs)
                    
                    # Step 2: Condition hand policy on arm actions
                    # Concatenate arm actions to hand observations
                    hand_actor_obs_conditioned = torch.cat([hand_actor_obs, arm_actions], dim=-1)
                    hand_critic_obs_conditioned = torch.cat([hand_critic_obs, arm_actions], dim=-1)
                    
                    # Step 3: Get hand actions given conditioned observations
                    hand_actions = self.hand_alg.act(hand_actor_obs_conditioned, hand_critic_obs_conditioned)
                    
                    # Step 4: Concatenate arm and hand actions for environment
                    actions = torch.cat((arm_actions, hand_actions), dim=1)

                    obs_dict, rewards, dones, terminated, time_outs, infos = self.env.step(actions)

                    arm_actor_obs, arm_critic_obs = obs_dict["arm_actor"], obs_dict["arm_critic"]
                    hand_actor_obs, hand_critic_obs = obs_dict["hand_actor"], obs_dict["hand_critic"]

                    self.arm_alg.process_env_step(rewards["arm"], dones, time_outs)
                    self.hand_alg.process_env_step(rewards["hand"], dones, time_outs)

                    if self.log_dir is not None:
                        # * Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum["arm"] += rewards["arm"]
                        cur_reward_sum["hand"] += rewards["hand"]
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer["arm"].extend(cur_reward_sum["arm"][new_ids][:, 0].cpu().numpy().tolist())
                        rewbuffer["hand"].extend(cur_reward_sum["hand"][new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum["arm"][new_ids] = 0
                        cur_reward_sum["hand"][new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # * Learning step
                start = stop
                # Condition last hand observations with current arm actions for computing returns
                # We need to condition the NEW observation (after last step) for computing returns.
                # We need the arm action from the CURRENT state (s_{N+1}), not the previous one.
                # Compute arm action from current arm observation for proper conditioning
                current_arm_actions = self.arm_alg.actor_critic.act_inference(arm_actor_obs)
                hand_critic_obs_conditioned = torch.cat([hand_critic_obs, current_arm_actions], dim=-1)
                
                self.arm_alg.compute_returns(arm_critic_obs)
                self.hand_alg.compute_returns(hand_critic_obs_conditioned)

            arm_mean_value_loss, arm_mean_surrogate_loss = self.arm_alg.update()
            hand_mean_value_loss, hand_mean_surrogate_loss = self.hand_alg.update()

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
        arm_mean_std = self.arm_alg.actor_critic.std.mean()
        hand_mean_std = self.hand_alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        if self.cfg["enable_logging"]:
            self.writer.add_scalar("Loss/arm/value_function", locs["arm_mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/arm/surrogate", locs["arm_mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/hand/value_function", locs["hand_mean_value_loss"], locs["it"])
            self.writer.add_scalar("Loss/hand/surrogate", locs["hand_mean_surrogate_loss"], locs["it"])
            self.writer.add_scalar("Loss/learning_rate", self.arm_alg.learning_rate, locs["it"])
            self.writer.add_scalar("Policy/arm/mean_noise_std", arm_mean_std.item(), locs["it"])
            self.writer.add_scalar("Policy/hand/mean_noise_std", hand_mean_std.item(), locs["it"])
            self.writer.add_scalar("Policy/arm/advantage_variance", self.arm_alg.storage.raw_advantages.var(), locs["it"])
            self.writer.add_scalar("Policy/hand/advantage_variance", self.hand_alg.storage.raw_advantages.var(), locs["it"])
            self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
            self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
            if len(locs["rewbuffer"]["arm"]) > 0:
                self.writer.add_scalar("Train/mean_reward/arm", statistics.mean(locs["rewbuffer"]["arm"]), locs["it"])
                self.writer.add_scalar("Train/mean_reward/hand", statistics.mean(locs["rewbuffer"]["hand"]), locs["it"])
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
                if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                    self.writer.add_scalar("Train/mean_reward/arm/time", statistics.mean(locs["rewbuffer"]["arm"]), self.tot_time)
                    self.writer.add_scalar("Train/mean_reward/hand/time", statistics.mean(locs["rewbuffer"]["hand"]), self.tot_time)
                    self.writer.add_scalar(
                        "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                    )

            self.log_buffer["Policy/arm/advantage_variance"].append(self.arm_alg.storage.raw_advantages.var().item())
            self.log_buffer["Policy/hand/advantage_variance"].append(self.hand_alg.storage.raw_advantages.var().item())
            self.log_buffer["Train/mean_reward/arm"].append(statistics.mean(locs["rewbuffer"]["arm"]))
            self.log_buffer["Train/mean_reward/hand"].append(statistics.mean(locs["rewbuffer"]["hand"]))

        # Video recording for wandb
        if self.cfg["enable_logging"] and self.logger_type == "wandb":
            self.writer.update_video_files(log_name="Video", fps=30)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]["arm"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss/arm:':>{pad}} {locs['arm_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/arm:':>{pad}} {locs['arm_mean_surrogate_loss']:.4f}\n"""
                f"""{'Value function loss/hand:':>{pad}} {locs['hand_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/hand:':>{pad}} {locs['hand_mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std/arm:':>{pad}} {arm_mean_std.item():.2f}\n"""
                f"""{'Mean action noise std/hand:':>{pad}} {hand_mean_std.item():.2f}\n"""
                f"""{'Mean reward/arm:':>{pad}} {statistics.mean(locs['rewbuffer']["arm"]):.2f}\n"""
                f"""{'Mean reward/hand:':>{pad}} {statistics.mean(locs['rewbuffer']["hand"]):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss/arm:':>{pad}} {locs['arm_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/arm:':>{pad}} {locs['arm_mean_surrogate_loss']:.4f}\n"""
                f"""{'Value function loss/hand:':>{pad}} {locs['hand_mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss/hand:':>{pad}} {locs['hand_mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std/arm:':>{pad}} {arm_mean_std.item():.2f}\n"""
                f"""{'Mean action noise std/hand:':>{pad}} {hand_mean_std.item():.2f}\n"""
            )

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
            "arm_model_state_dict": self.arm_alg.actor_critic.state_dict(),
            "arm_optimizer_state_dict": self.arm_alg.optimizer.state_dict(),
            "hand_model_state_dict": self.hand_alg.actor_critic.state_dict(),
            "hand_optimizer_state_dict": self.hand_alg.optimizer.state_dict(),
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

        # # 2. Mirror copy inside scripts/plotting/data/<experiment_name>/
        # with data_dir_path.open("wb") as f:
        #     pickle.dump(dict(self.log_buffer), f)

    def load(self, path, load_modular: bool = True, load_optimizer: bool = True, only_arm: bool = False):
        try:
            loaded_dict = torch.load(path)
        except:
            import sys
            sys.modules['learning'] = sys.modules['rsl_rl']
            sys.modules['learning.storage'] = sys.modules['rsl_rl.storage']
            loaded_dict = torch.load(path)

        if load_modular:
            if only_arm:
                self.arm_alg.actor_critic.load_state_dict(loaded_dict["arm_model_state_dict"])
                if load_optimizer:
                    self.arm_alg.optimizer.load_state_dict(loaded_dict["arm_optimizer_state_dict"])
            else:
                self.arm_alg.actor_critic.load_state_dict(loaded_dict["arm_model_state_dict"])
                self.hand_alg.actor_critic.load_state_dict(loaded_dict["hand_model_state_dict"])
                if load_optimizer:
                    self.arm_alg.optimizer.load_state_dict(loaded_dict["arm_optimizer_state_dict"])
                    self.hand_alg.optimizer.load_state_dict(loaded_dict["hand_optimizer_state_dict"])
        else:
            self.arm_alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
            if load_optimizer:
                self.arm_alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.arm_alg.actor_critic.to(device)
            self.hand_alg.actor_critic.to(device)
        arm_actor_inference = self.arm_alg.actor_critic.act_inference
        hand_actor_inference = self.hand_alg.actor_critic.act_inference

        def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            arm_actor_obs, hand_actor_obs = obs_dict["arm_actor"], obs_dict["hand_actor"]
            
            # Step 1: Get arm actions
            arm_actions = arm_actor_inference(arm_actor_obs)
            
            # Step 2: Condition hand policy on arm actions
            hand_actor_obs_conditioned = torch.cat([hand_actor_obs, arm_actions], dim=-1)
            
            # Step 3: Get hand actions given conditioned observations
            hand_actions = hand_actor_inference(hand_actor_obs_conditioned)
            
            # Step 4: Concatenate arm and hand actions
            actions = torch.cat((arm_actions, hand_actions), dim=1)
            return actions

        return policy
    def train_mode(self):
        self.arm_alg.actor_critic.train()
        self.hand_alg.actor_critic.train()

    def eval_mode(self):
        self.arm_alg.actor_critic.eval()
        self.hand_alg.actor_critic.eval()

    def export(self, path, model_name):
        self.arm_alg.actor_critic.export_policy(path, model_name + "_arm")
        self.hand_alg.actor_critic.export_policy(path, model_name + "_hand")


