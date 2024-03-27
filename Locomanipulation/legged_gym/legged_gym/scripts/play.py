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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.terrain.tot_cols = 600
    env_cfg.terrain.tot_rows = 600
    env_cfg.terrain.transform_x = - env_cfg.terrain.tot_cols * env_cfg.terrain.horizontal_scale / 2
    env_cfg.terrain.transform_y = - env_cfg.terrain.tot_rows * env_cfg.terrain.horizontal_scale / 2
    env_cfg.terrain.zScale = 0.0
    
    # env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.episode_length_s = 1000
    
    ################################
    env_cfg.goal_ee.key_ctrl = True
    ################################
    
    env_cfg.commands.lin_vel_x_schedule =  [0, 1]
    env_cfg.commands.ang_vel_yaw_schedule = [0, 1]

    env_cfg.commands.ranges.final_lin_vel_x = [0.8, 0.8] # Forward
    env_cfg.commands.ranges.final_ang_vel_yaw = [0.0, 0.0] # Foward
    
    # env_cfg.commands.ranges.final_lin_vel_x = [1.5, 1.5] # Fast Forward
    # env_cfg.commands.ranges.final_ang_vel_yaw = [0.0, 0.0] # Fast Foward
    
    # env_cfg.commands.ranges.final_lin_vel_x = [1.0, 1.0] # Forward and turn
    # env_cfg.commands.ranges.final_ang_vel_yaw = [0.8, 0.8] # Foward and turn
    
    # env_cfg.commands.ranges.final_lin_vel_x = [0.0, 0.0] # Stance
    # env_cfg.commands.ranges.final_ang_vel_yaw = [0.0, 0.0] # Stance
    
    # env_cfg.commands.ranges.final_lin_vel_x = [0.5, 0.5] # Rotate
    # env_cfg.commands.ranges.final_ang_vel_yaw = [0.8, 0.8] # Rotate

    env_cfg.goal_ee.l_schedule = [0, 1]
    env_cfg.goal_ee.p_schedule = [0, 1]
    env_cfg.goal_ee.y_schedule = [0, 1]
    env_cfg.goal_ee.arm_action_scale_schedule = [0, 1]
    env_cfg.goal_ee.tracking_ee_reward_schedule = [0, 1]
    env_cfg.goal_ee.underground_limit = -0.57  # -0.54 -0.4
    env_cfg.curriculum.mass_curriculum = False
    # env_cfg.goal_ee.ranges.final_pos_l = [0.6, 0.8]
    # env_cfg.goal_ee.ranges.final_pos_p = [-1.5 * np.pi / 5, 1.5 * np.pi / 5]
    # env_cfg.goal_ee.ranges.final_pos_y = [0, 0]
    
    # env_cfg.goal_ee.traj_time = [1.5, 1.5]
    env_cfg.goal_ee.test = False
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.65, 0.65]
    # env_cfg.goal_ee.ranges.final_pos_p = [1.5 * np.pi / 5, 1.5 * np.pi / 5]
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.65, 0.65]
    # env_cfg.goal_ee.ranges.final_pos_p = [-1.5 * np.pi / 5, 1.5 * np.pi / 5]
    # env_cfg.goal_ee.ranges.final_pos_y = [0, 0]
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.25, 0.25]
    # env_cfg.goal_ee.ranges.final_pos_p = [1.5 * np.pi / 5, 1.5 * np.pi / 5]
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.25, 0.25]
    # env_cfg.goal_ee.ranges.final_pos_p = [-1.5 * np.pi / 5, -1.5 * np.pi / 5]
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.65, 0.65]
    # env_cfg.goal_ee.ranges.final_pos_p = [0, 0]
    # env_cfg.goal_ee.ranges.fianl_pos_y = [-3 / 5 * np.pi, 3 / 5 * np.pi]
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.45, 0.45]
    # env_cfg.goal_ee.ranges.final_pos_p = [0, 0]
    # env_cfg.goal_ee.ranges.final_pos_y = [0, 0]
    
    # env_cfg.goal_ee.ranges.final_pos_l = [0.6, 0.6]
    # env_cfg.goal_ee.ranges.final_pos_p = [1.5 * np.pi / 5, 1.5 * np.pi / 5]
    # env_cfg.goal_ee.ranges.final_pos_y = [0, 0]
    
    env_cfg.goal_ee.ranges.init_pos_l = env_cfg.goal_ee.ranges.final_pos_l
    env_cfg.goal_ee.ranges.init_pos_p = env_cfg.goal_ee.ranges.final_pos_p
    env_cfg.goal_ee.ranges.init_pos_y = env_cfg.goal_ee.ranges.final_pos_y

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device, stochastic=args.stochastic)

    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    if SAVE_ACTOR_HIST_ENCODER:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        model_file = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
        model_name = model_file.split('/')[-1].split('.')[0]
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run, 'exported')
        os.makedirs(path, exist_ok=True)
        torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), path + '/' + model_name + '_actor.pt')
        print('Saved actor to: ', path + '/' + model_name + '_actor.pt')
    
    if args.use_jit:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, 'traced', args.load_run+"_actor_jit.pt")
        print("Loading jit for policy: ", path)
        policy = torch.jit.load(path, map_location=ppo_runner.device)
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, 'traced', args.load_run+"_hist_encoder_jit.pt")
        history_encoder = torch.jit.load(path, map_location=ppo_runner.device)
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 13 # which joint is used for logging 13: waist, 14: elbow
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    env.update_command_curriculum()
    env.reset()
    for i in range(100*int(env.max_episode_length)):
        start_time = time.time()
        if args.use_jit:
            latent = history_encoder(obs[:, env.cfg.env.num_proprio+env.cfg.env.num_priv:])
            actions = policy(torch.cat((obs[:, :env.cfg.env.num_proprio], latent), dim=1))
        else:
            actions = policy(obs.detach(), hist_encoding=False)
            # actions = policy(obs.detach(), hist_encoding=True)
        obs, _, rews, arm_rews, dones, infos = env.step(actions.detach())
        # input()
        if i % 15 == 0:
            command_detached = env.commands[0].detach().cpu().numpy()
            print('command: ', f'{command_detached[0]:.2f}', f'{command_detached[1]:.2f}', f'{command_detached[2]:.2f}')
        if i % 15 == 0:
            pass
            curr_ee_goal_sphere = env.curr_ee_goal_sphere[0].detach().cpu().numpy()
            ee_curr_orn_euler = env.ee_curr_orn_euler[0].detach().cpu().numpy()
            # print('\taction: ', actions.detach().cpu().numpy()[0, :])
            print('curr speed: ', f'{env.base_lin_vel[0, 0].detach().cpu().numpy():.2f}', f'{env.base_lin_vel[0, 1].detach().cpu().numpy():.2f}', f'{env.base_ang_vel[0, 2].detach().cpu().numpy():.2f}')
            print('curr_ee_goal_sphere: ', f'{curr_ee_goal_sphere[0]:.2f}', f'{curr_ee_goal_sphere[1]:.2f}', f'{curr_ee_goal_sphere[2]:.2f}')
            print('ee_curr_orn_euler: ', f'{ee_curr_orn_euler[0]:.2f}', f'{ee_curr_orn_euler[1]:.2f}', f'{ee_curr_orn_euler[2]:.2f}')
            # r, p = env.get_body_orientation() 
            # print('body roll: ', f'{r[0]:.2f}')
            # print('body pitch: ', f'{p[0]:.2f}')
        # print('\ttorque square: ', torch.sum(torch.square(env.torques[robot_index, :12])))
        # print('\tjoint velocity: ', f'{env.dof_vel[robot_index, joint_index].item(): .2f}')
        # print('\tfoot_contacts: ', f'{env.contact[robot_index]}')
        # print('\tair time: ', env.air_time_)
        # print('\tclearance: ', env.clearance_)
        if RECORD_FRAMES:
            if i % 1:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * np.array(env.cfg.control.action_scale),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        stop_time = time.time()

        duration = stop_time - start_time
        time.sleep(max(0.02 - duration, 0))

if __name__ == '__main__':
    EXPORT_POLICY = False
    SAVE_ACTOR_HIST_ENCODER = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args(test=True)
    play(args)
