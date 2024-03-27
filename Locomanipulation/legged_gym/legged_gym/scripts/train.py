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
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import wandb

def train(args):
    wandb.init(project='manip-loco_test', name=str(args.exptid) + '_' + args.run_name)
    wandb.save(LEGGED_GYM_ENVS_DIR + "/widowGo1/widowGo1_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/widowGo1/widowGo1.py", policy="now")
    wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/modules/actor_critic.py", policy="now")
    wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/algorithms/ppo.py", policy="now")
    wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/runners/on_policy_runner.py", policy="now")
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs =  2000 # max 4000
    env_cfg.terrain.tot_cols = 3000
    env_cfg.terrain.tot_rows = 1000
    # env_cfg.terrain.tot_cols = 400
    # env_cfg.terrain.tot_rows = 400
    env_cfg.terrain.transform_x = - env_cfg.terrain.tot_cols * env_cfg.terrain.horizontal_scale / 2
    env_cfg.terrain.transform_y = - env_cfg.terrain.tot_rows * env_cfg.terrain.horizontal_scale / 2
    env_cfg.terrain.zScale = 0.0

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # env_cfg.commands.ranges.init_lin_vel_x = env_cfg.commands.ranges.final_lin_vel_x
    # env_cfg.commands.ranges.init_lin_vel_y = env_cfg.commands.ranges.final_lin_vel_y
    # env_cfg.commands.ranges.init_ang_vel_yaw = env_cfg.commands.ranges.final_ang_vel_yaw

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
