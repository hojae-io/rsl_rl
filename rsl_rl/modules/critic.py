import torch
import torch.nn as nn
from .utils import create_MLP, weights_init_
from .utils import EmpiricalNormalization

class Critic(nn.Module):
    def __init__(self,
                 num_obs,
                 hidden_dims,
                 activation="elu",
                 normalize_obs=False,
                 custom_initialization=False,
                 **kwargs):

        if kwargs:
            print("Critic.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation)

        if normalize_obs:
            self.obs_normalizer = EmpiricalNormalization(num_obs, until=1.0e8)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            
        if custom_initialization:
            self.apply(weights_init_)

    def evaluate(self, critic_observations, actions=None):
        if actions is None: 
            # Critic is V(s)-function estimator
            return self.NN(self.norm_obs(critic_observations))
        else: 
            # Critic is Q(s,a)-function estimaotr
            concat_input = torch.cat((critic_observations, actions), dim=1)
            return self.NN(concat_input)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.obs_normalizer(observation)

    def freeze_parameters(self):
        for parameters in self.NN.parameters():
            parameters.requires_grad = False

    def update_parameters(self, src_model: 'Critic', polyak: float):
        with torch.inference_mode():
            for parameters, src_parameters in zip(self.NN.parameters(), src_model.NN.parameters()):
                parameters.data.mul_(1 - polyak)
                parameters.data.add_(polyak * src_parameters.data)
