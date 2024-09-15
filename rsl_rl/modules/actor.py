import os
import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import create_MLP, weights_init_
from .utils import EmpiricalNormalization

class Actor(nn.Module):
    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims,
                 activation="elu",
                 init_noise_std=1.0,
                 normalize_obs=False,
                 log_std_bounds=None,
                 actions_limits=None,
                 custom_initialization=False,
                 **kwargs):

        if kwargs:
            print("Actor.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        if normalize_obs:
            self.obs_normalizer = EmpiricalNormalization(num_obs, until=1.0e8)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization

        self.mean_NN = create_MLP(num_obs, num_actions, hidden_dims, activation)
        self.log_std_NN = None

        # Action noise
        if log_std_bounds is not None:
            self.log_std_min, self.log_std_max = log_std_bounds
            self.log_std_NN = create_MLP(num_obs, num_actions, hidden_dims, activation)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.distribution = None

        if actions_limits is not None:
            self.actions_min, self.actions_max = actions_limits
            self.actions_range_center = (self.actions_max + self.actions_min) / 2
            self.actions_range_radius = (self.actions_max - self.actions_min) / 2

        # disable args validation for speedup
        Normal.set_default_validate_args = False
        if custom_initialization:
            torch.nn.init.constant_(self.mean_NN[-1].weight, 0)
            torch.nn.init.constant_(self.mean_NN[-1].bias, 0)
            # self.apply(weights_init_) # xavier_uniform initialization


    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.mean_NN(self.norm_obs(observations))
        if self.log_std_NN is None:
            self.distribution = Normal(mean, mean*0. + self.std)
        else: # TODO: Implement s.t. mean & log_std shares parameters only last layer is different!
            log_std = self.log_std_NN(observations)
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
            self.std = torch.exp(log_std)
            self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def ract(self, observations):
        """ Sample with reparametrization trick """
        self.update_distribution(observations)
        return self.distribution.rsample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_scaled_ractions_and_log_prob(self, observations, only_actions=False):
        """ Get scaled actions using reparametrization trick and their log probability
            Implemented solely for SAC """ 
        self.update_distribution(observations)
        actions = self.distribution.rsample()
        actions_normalized = torch.tanh(actions)
        actions_scaled = (self.actions_range_center + self.actions_range_radius * actions_normalized)
        
        if only_actions:
            return actions_scaled 
        else:
            actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1) - \
                               torch.log(1.0 - actions_normalized.pow(2) + 1e-6).sum(-1)
            return actions_scaled, actions_log_prob

    def act_inference(self, observations):
        actions_mean = self.mean_NN(self.norm_obs(observations))
        return actions_mean

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.obs_normalizer(observation)
        
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path_TS = os.path.join(path, 'policy.pt') # TorchScript path
        path_onnx = os.path.join(path, 'policy.onnx') # ONNX path

        class NormalizedActor(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer
            def forward(self, obs):
                obs = self.obs_normalizer(obs)
                return self.actor(obs).squeeze(0)
        model = NormalizedActor(copy.deepcopy(self.mean_NN), copy.deepcopy(self.obs_normalizer)).to('cpu')
        
        dummy_input = torch.rand(self.mean_NN[0].in_features,)
        model_traced = torch.jit.trace(model, dummy_input)
        torch.jit.save(model_traced, path_TS)
        torch.onnx.export(model_traced, dummy_input, path_onnx)
