import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from icecream import ic


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs):
        self.action_mean = self.architecture(obs).cpu().numpy()
        actions, log_prob = self.distribution.sample(self.action_mean)
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

    @property
    def priv_shape(self):
        return self.architecture.priv_shape

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class Actor_MLP(nn.Module):
    def __init__(self, hidden_dims, leg_hidden_dims, arm_hidden_dims, priv_encoder_dims, activation_fn,
                 num_priv, num_prop, num_leg_actions, num_arm_actions):
        super(Actor_MLP, self).__init__()
        self.activation_fn = activation_fn
        self.num_priv = num_priv
        self.num_prop = num_prop
        self.input_shape = [self.num_priv + self.num_prop]
        self.output_shape = [num_leg_actions + num_arm_actions]
        self.priv_shape = [self.num_priv]

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
            priv_encoder_layers.append(self.activation_fn())
            for l in range(len(priv_encoder_dims) - 1):
                # if l == len(priv_encoder_dims) - 1:
                #     priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], num_actions))
                #     # priv_encoder_layers.append(nn.Tanh())
                # else:
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                priv_encoder_layers.append(self.activation_fn())
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv

            # Policy
        if len(hidden_dims) > 0:
            actor_layers = []
            actor_layers.append(nn.Linear(num_prop + priv_encoder_output_dim, hidden_dims[0]))
            actor_layers.append(self.activation_fn())
            for l in range(len(hidden_dims) - 1):
                    # if l == len(actor_hidden_dims) - 1:
                    #     actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                    #     # actor_layers.append(nn.Tanh())
                    # else:
                actor_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                actor_layers.append(self.activation_fn())
            self.actor_backbone = nn.Sequential(*actor_layers)
            actor_backbone_output_dim = hidden_dims[-1]
        else:
            self.actor_backbone = nn.Identity()
            actor_backbone_output_dim = num_prop + priv_encoder_output_dim

        actor_leg_layers = []
        actor_leg_layers.append(nn.Linear(actor_backbone_output_dim, leg_hidden_dims[0]))
        actor_leg_layers.append(self.activation_fn())
        for l in range(len(leg_hidden_dims)):
            if l == len(leg_hidden_dims) - 1:
                actor_leg_layers.append(nn.Linear(leg_hidden_dims[l], num_leg_actions))
                actor_leg_layers.append(nn.Tanh())
            else:
                actor_leg_layers.append(nn.Linear(leg_hidden_dims[l], leg_hidden_dims[l + 1]))
                actor_leg_layers.append(self.activation_fn())
        self.actor_leg_control_head = nn.Sequential(*actor_leg_layers)

        actor_arm_layers = []
        actor_arm_layers.append(nn.Linear(actor_backbone_output_dim, arm_hidden_dims[0]))
        actor_arm_layers.append(self.activation_fn())
        for l in range(len(arm_hidden_dims)):
            if l == len(arm_hidden_dims) - 1:
                actor_arm_layers.append(nn.Linear(arm_hidden_dims[l], num_arm_actions))
                actor_arm_layers.append(nn.Tanh())
            else:
                actor_arm_layers.append(nn.Linear(arm_hidden_dims[l], arm_hidden_dims[l + 1]))
                actor_arm_layers.append(self.activation_fn())
        self.actor_arm_control_head = nn.Sequential(*actor_arm_layers)

    def forward(self, obs, hist_encoding=False):
        obs_prop = obs[:, :self.num_prop]
        # if hist_encoding:
        #     latent = self.infer_hist_latent(obs)
        # else:
        #     latent = self.infer_priv_latent(obs)
        latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop, latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        leg_output = self.actor_leg_control_head(backbone_output)
        arm_output = self.actor_arm_control_head(backbone_output)

        return torch.cat([leg_output, arm_output], dim=-1)

    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop: self.num_prop + self.num_priv]
        return self.priv_encoder(priv)

        # def infer_hist_latent(self, obs):
        #     hist = obs[:, -self.num_hist*self.num_prop:]
        #     return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

class Critic_MLP(nn.Module):
    def __init__(self, hidden_dims, leg_hidden_dims, arm_hidden_dims, activation_fn,
                 num_priv, num_prop):
        super(Critic_MLP, self).__init__()
        self.activation_fn = activation_fn
        self.num_priv = num_priv
        self.num_prop = num_prop
        self.input_shape = [num_priv + num_prop]

        if len(hidden_dims) > 0:
            critic_layers = []
            critic_layers.append(nn.Linear(num_priv + num_prop, hidden_dims[0]))
            critic_layers.append(self.activation_fn())
            for l in range(len(hidden_dims) - 1):
                # if l == len(critic_hidden_dims) - 1:
                #     critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                # else:
                critic_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                critic_layers.append(self.activation_fn())
            self.critic_backbone = nn.Sequential(*critic_layers)
            critic_backbone_output_dim = hidden_dims[-1]
        else:
            self.critic_backbone = nn.Identity()
            critic_backbone_output_dim = num_priv + num_prop

        critic_leg_layers = []
        critic_leg_layers.append(nn.Linear(critic_backbone_output_dim, leg_hidden_dims[0]))
        critic_leg_layers.append(self.activation_fn())
        for l in range(len(leg_hidden_dims)):
            if l == len(leg_hidden_dims) - 1:
                critic_leg_layers.append(nn.Linear(leg_hidden_dims[l], 1))
            else:
                critic_leg_layers.append(nn.Linear(leg_hidden_dims[l], leg_hidden_dims[l + 1]))
                critic_leg_layers.append(self.activation_fn())
        self.critic_leg_control_head = nn.Sequential(*critic_leg_layers)

        critic_arm_layers = []
        critic_arm_layers.append(nn.Linear(critic_backbone_output_dim, arm_hidden_dims[0]))
        critic_arm_layers.append(self.activation_fn())
        for l in range(len(arm_hidden_dims)):
            if l == len(arm_hidden_dims) - 1:
                critic_arm_layers.append(nn.Linear(arm_hidden_dims[l], 1))
            else:
                critic_arm_layers.append(nn.Linear(arm_hidden_dims[l], arm_hidden_dims[l + 1]))
                critic_arm_layers.append(self.activation_fn())
        self.critic_arm_control_head = nn.Sequential(*critic_arm_layers)

    def forward(self, obs):
        prop_and_priv = obs[:, :self.num_prop + self.num_priv]
        backbone_output = self.critic_backbone(prop_and_priv)
        leg_output = self.critic_leg_control_head(backbone_output)
        arm_output = self.critic_arm_control_head(backbone_output)
        return torch.cat([leg_output, arm_output], dim=-1)

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, num_leg_actions, num_arm_actions, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.num_leg_actions = num_leg_actions
        self.num_arm_actions = num_arm_actions
        self.dim = num_leg_actions + num_arm_actions
        self.std = nn.Parameter(init_std * torch.ones(self.dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, num_leg_actions+num_arm_actions], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        log_prob = distribution.log_prob(outputs)
        leg_log_prob_sum = log_prob[:, :self.num_leg_actions].sum(dim=1, keepdims=True)
        arm_log_prob_sum = log_prob[:, self.num_leg_actions:].sum(dim=1, keepdims=True)
        actions_log_prob = torch.cat([leg_log_prob_sum, arm_log_prob_sum], dim=-1)

        entropy_ = distribution.entropy()
        leg_entropy_sum = entropy_[:, :self.num_leg_actions].sum(dim=1, keepdims=True)
        arm_entropy_sum = entropy_[:, self.num_leg_actions:].sum(dim=1, keepdims=True)
        entropy = torch.cat([leg_entropy_sum, arm_entropy_sum], dim=-1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

