# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import copy, math
import torch.nn.functional as F

# History Encoder
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )


    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


# Transformer Encoder
class Encoder(nn.Module):

    def __init__(self, n_layer, d_embed, d_model, h, d_ff, norm_eps = 1e-5):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.d_embed = d_embed
        self.d_model = d_model
        self.h = h
        
        self.norm = nn.LayerNorm(d_embed, eps = norm_eps)
        
        self.self_attention = MultiHeadAttentionLayer(d_model = d_model, h = h, 
                                            qkv_fc = nn.Linear(d_embed, d_model),
                                            out_fc = nn.Linear(d_model, d_embed))
        
        self.position_ff = PositionWiseFeedForwardLayer(fc1 = nn.Linear(d_embed, d_ff), 
                                            fc2 = nn.Linear(d_ff, d_embed))
        
        self.layers = nn.ModuleList([EncoderBlock(self_attention=self.self_attention,
                                                  position_ff=self.position_ff,
                                                  norm=self.norm) for _ in range(self.n_layer)])
    

    def forward(self, src: torch.Tensor, src_mask):
        out1 = src[:, :18].view(-1, 6, 3)
        out2 = src[:, 18:36].view(-1, 6, 3)
        out3 = src[:, 36:54].view(-1, 6, 3)
        out = torch.cat([out1, out2, out3], dim = 2) # [num_envs, 6, 9]
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)
        return out.view(-1, 54)
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention   
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)


    def forward(self, src, src_mask):
        out = src
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)
        return out
    

class ResidualConnectionLayer(nn.Module):

    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=dr_rate)


    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out
    

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc              # (d_model, d_embed)
        self.dropout = nn.Dropout(p=dr_rate)


    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
        return out


    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc): # (n_batch, seq_len, d_embed)
            out = fc(x)       # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)       # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out
    

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, fc1, fc2, dr_rate=0):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1   # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2 # (d_ff, d_embed)


    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        priv_encoder_dims=[64, 20],
                        activation='elu',
                        init_std=1,
                        **kwargs):
        # if kwargs:
        #     print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        # self.dual_heads = kwargs['dual_heads']
        leg_control_head_hidden_dims = kwargs['leg_control_head_hidden_dims']
        arm_control_head_hidden_dims = kwargs['arm_control_head_hidden_dims']
        self.num_leg_actions = kwargs['num_leg_actions']
        self.num_arm_actions = kwargs['num_arm_actions']
        adaptive_arm_gains = kwargs['adaptive_arm_gains']
        adaptive_arm_gains_scale = kwargs['adaptive_arm_gains_scale']
        num_priv = kwargs['num_priv']
        num_hist = kwargs['num_hist']
        num_prop = kwargs['num_prop']
        include_history = kwargs['include_history']
        if adaptive_arm_gains:
            self.num_arm_actions *= 2

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        class Actor(nn.Module):
            def __init__(self, mlp_input_dim_a, actor_hidden_dims, activation, leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
                num_leg_actions, num_arm_actions, adaptive_arm_gains, adaptive_arm_gains_scale,
                num_priv, num_hist, num_prop, priv_encoder_dims, include_history):
                super().__init__()
                self.adaptive_arm_gains = adaptive_arm_gains
                self.adaptive_arm_gains_scale = adaptive_arm_gains_scale
                self.num_arm_actions = num_arm_actions

                # self.encoder = Encoder(n_layer= 3, d_embed= 9, d_model = 9, h = 3, d_ff = 6)
                
                # Policy
                if len(priv_encoder_dims) > 0:
                    priv_encoder_layers = []
                    priv_encoder_layers.append(nn.Linear(num_priv, priv_encoder_dims[0]))
                    priv_encoder_layers.append(activation)
                    for l in range(len(priv_encoder_dims) - 1):
                        # if l == len(priv_encoder_dims) - 1:
                        #     priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], num_actions))
                        #     # priv_encoder_layers.append(nn.Tanh())
                        # else:
                        priv_encoder_layers.append(nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1]))
                        priv_encoder_layers.append(activation)
                    self.priv_encoder = nn.Sequential(*priv_encoder_layers)
                    priv_encoder_output_dim = priv_encoder_dims[-1]
                else:
                    self.priv_encoder = nn.Identity()
                    priv_encoder_output_dim = num_priv

                self.num_priv = num_priv
                self.num_hist = num_hist
                self.num_prop = num_prop

                # Priv Encoder
                # encoder_dim = 8
                # self.priv_encoder =  nn.Sequential(*[
                #                         nn.Linear(num_priv, 256), activation,
                #                         nn.Linear(256, 128), activation,
                #                         nn.Linear(128, encoder_dim), 
                #                         # nn.Tanh()
                #                         nn.LeakyReLU()
                #                     ])
                if include_history:
                    self.history_encoder = StateHistoryEncoder(activation, mlp_input_dim_a, num_hist, priv_encoder_output_dim)

                # Policy
                if len(actor_hidden_dims) > 0:
                    actor_layers = []
                    actor_layers.append(nn.Linear(mlp_input_dim_a + priv_encoder_output_dim, actor_hidden_dims[0]))
                    actor_layers.append(activation)
                    for l in range(len(actor_hidden_dims) - 1):
                        # if l == len(actor_hidden_dims) - 1:
                        #     actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                        #     # actor_layers.append(nn.Tanh())
                        # else:
                        actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                        actor_layers.append(activation)
                    self.actor_backbone = nn.Sequential(*actor_layers)
                    actor_backbone_output_dim = actor_hidden_dims[-1]
                else:
                    self.actor_backbone = nn.Identity()
                    actor_backbone_output_dim = mlp_input_dim_a + priv_encoder_output_dim

                actor_leg_layers = []
                actor_leg_layers.append(nn.Linear(actor_backbone_output_dim, leg_control_head_hidden_dims[0]))
                actor_leg_layers.append(activation)
                for l in range(len(leg_control_head_hidden_dims)):
                    if l == len(leg_control_head_hidden_dims) - 1:
                        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], num_leg_actions))
                        actor_leg_layers.append(nn.Tanh())
                    else:
                        actor_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                        actor_leg_layers.append(activation)
                self.actor_leg_control_head = nn.Sequential(*actor_leg_layers)

                actor_arm_layers = []
                actor_arm_layers.append(nn.Linear(actor_backbone_output_dim, arm_control_head_hidden_dims[0]))
                actor_arm_layers.append(activation)
                for l in range(len(arm_control_head_hidden_dims)):
                    if l == len(arm_control_head_hidden_dims) - 1:
                        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], num_arm_actions))
                        actor_arm_layers.append(nn.Tanh())
                    else:
                        actor_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                        actor_arm_layers.append(activation)
                self.actor_arm_control_head = nn.Sequential(*actor_arm_layers)
            
            def forward(self, obs, hist_encoding=False):
                # obs_joint = obs[:, :54] # 18 * 3
                # obs[:, :54] = self.encoder(obs_joint, None)
                obs_prop = obs[:, :self.num_prop]
                if hist_encoding:
                    latent = self.infer_hist_latent(obs)
                else:
                    latent = self.infer_priv_latent(obs)
                backbone_input = torch.cat([obs_prop, latent], dim=1)
                backbone_output = self.actor_backbone(backbone_input)
                leg_output = self.actor_leg_control_head(backbone_output)
                arm_output = self.actor_arm_control_head(backbone_output)
                if self.adaptive_arm_gains:
                    gains = self.adaptive_arm_gains_scale * (arm_output[:, self.num_arm_actions // 2:])
                    arm_output = torch.cat([arm_output[:, :self.num_arm_actions // 2], gains], dim=-1)
                return torch.cat([leg_output, arm_output], dim=-1)
            
            def infer_priv_latent(self, obs):
                priv = obs[:, self.num_prop: self.num_prop + self.num_priv]
                return self.priv_encoder(priv)
            
            def infer_hist_latent(self, obs):
                hist = obs[:, -self.num_hist*self.num_prop:]
                return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
            
        self.actor = Actor(mlp_input_dim_a, actor_hidden_dims, activation, leg_control_head_hidden_dims, arm_control_head_hidden_dims, \
            self.num_leg_actions, self.num_arm_actions, adaptive_arm_gains, adaptive_arm_gains_scale,
            num_priv, num_hist, num_prop, priv_encoder_dims, include_history)


        # Value function
        class Critic(nn.Module):
            def __init__(self, mlp_input_dim_c, critic_hidden_dims, activation, leg_control_head_hidden_dims, arm_control_head_hidden_dims,
                         num_priv, num_hist, num_prop):
                super().__init__()

                self.num_priv = num_priv
                self.num_hist = num_hist
                self.num_prop = num_prop

                # Value
                if len(critic_hidden_dims) > 0:
                    critic_layers = []
                    critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
                    critic_layers.append(activation)
                    for l in range(len(critic_hidden_dims) - 1):
                        # if l == len(critic_hidden_dims) - 1:
                        #     critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                        # else:
                        critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                        critic_layers.append(activation)
                    self.critic_backbone = nn.Sequential(*critic_layers)
                    critic_backbone_output_dim = critic_hidden_dims[-1]
                else:
                    self.critic_backbone = nn.Identity()
                    critic_backbone_output_dim = mlp_input_dim_c

                critic_leg_layers = []
                critic_leg_layers.append(nn.Linear(critic_backbone_output_dim, leg_control_head_hidden_dims[0]))
                critic_leg_layers.append(activation)
                for l in range(len(leg_control_head_hidden_dims)):
                    if l == len(leg_control_head_hidden_dims) - 1:
                        critic_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], 1))
                    else:
                        critic_leg_layers.append(nn.Linear(leg_control_head_hidden_dims[l], leg_control_head_hidden_dims[l + 1]))
                        critic_leg_layers.append(activation)
                self.critic_leg_control_head = nn.Sequential(*critic_leg_layers)

                critic_arm_layers = []
                critic_arm_layers.append(nn.Linear(critic_backbone_output_dim, arm_control_head_hidden_dims[0]))
                critic_arm_layers.append(activation)
                for l in range(len(arm_control_head_hidden_dims)):
                    if l == len(arm_control_head_hidden_dims) - 1:
                        critic_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], 1))
                    else:
                        critic_arm_layers.append(nn.Linear(arm_control_head_hidden_dims[l], arm_control_head_hidden_dims[l + 1]))
                        critic_arm_layers.append(activation)
                self.critic_arm_control_head = nn.Sequential(*critic_arm_layers)
            
            def forward(self, obs):
                prop_and_priv = obs[:, :self.num_prop + self.num_priv]
                backbone_output = self.critic_backbone(prop_and_priv)
                leg_output = self.critic_leg_control_head(backbone_output)
                arm_output = self.critic_arm_control_head(backbone_output)
                return torch.cat([leg_output, arm_output], dim=-1)

        self.critic = Critic(mlp_input_dim_c + num_priv, critic_hidden_dims, activation, leg_control_head_hidden_dims, arm_control_head_hidden_dims, 
                             num_priv, num_hist, num_prop)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(torch.tensor(init_std))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        entropy = self.distribution.entropy()
        leg_entropy_sum = entropy[:, :self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_entropy_sum = entropy[:, self.num_leg_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([leg_entropy_sum, arm_entropy_sum], dim=-1)

    def update_distribution(self, observations, hist_encoding):
        mean = self.actor(observations, hist_encoding)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, hist_encoding, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions)
        leg_log_prob_sum = log_prob[:, :self.num_leg_actions].sum(dim=-1, keepdim=True)
        arm_log_prob_sum = log_prob[:, self.num_leg_actions:].sum(dim=-1, keepdim=True)
        return torch.cat([leg_log_prob_sum, arm_log_prob_sum], dim=-1)

    def act_inference(self, observations, hist_encoding=False):
        actions_mean = self.actor(observations, hist_encoding)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
