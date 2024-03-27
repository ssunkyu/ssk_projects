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

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, quat_mul, quat_conjugate
from typing import Tuple, Optional

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def torch_wrap_to_pi_minuspi(angles):
    angles = angles % (2 * np.pi)
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor,
                  out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
    x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
    if out is None:
        out = torch.empty_like(q1)
    out[...] = torch.stack([
        x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
        -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
        x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
        -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2], dim=-1)
    return out

def quat_inverse(q: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = q.clone()
    out.copy_(q)
    out[..., 3] = -out[..., 3]
    return out

def cart2sphere(cart):
    sphere = torch.zeros_like(cart)
    sphere[:, 0] = torch.norm(cart, dim=-1)
    sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])
    sphere[:, 2] = torch.asin(cart[:, 1] / sphere[:, 0])
    return sphere

def sphere2cart(sphere):
    cart = torch.zeros_like(sphere)
    cart[:, 0] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.cos(sphere[:, 1])
    cart[:, 1] = sphere[:, 0] * torch.sin(sphere[:, 2])
    cart[:, 2] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.sin(sphere[:, 1])
    return cart

def torch_rand_sign(shape, device):
    return 2 * torch.randint(0, 2, shape, device=device) - 1

def quaternion_to_rotation_matrix(quat):
    # quat shape is [num_env, 4]
    x, y, z, w = quat.unbind(-1)
    Nq = w*w + x*x + y*y + z*z
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    rotation_matrix = torch.stack([
        1.0 - (yY + zZ), xY - wZ, xZ + wY,
        xY + wZ, 1.0 - (xX + zZ), yZ - wX,
        xZ - wY, yZ + wX, 1.0 - (xX + yY),
    ], dim=-1).reshape(-1, 3, 3)

    return rotation_matrix

def skew_symmetric_matrix(vector):
    zero = torch.zeros(vector.shape[0], 1, device=vector.device)
    matrix = torch.cat([
        zero, -vector[:, 2:3], vector[:, 1:2],
        vector[:, 2:3], zero, -vector[:, 0:1],
        -vector[:, 1:2], vector[:, 0:1], zero,
    ], dim=-1).reshape(-1, 3, 3)
    return matrix

def get_adjoint_matrix(p, quat):
    rotation_matrix_transpose = torch.transpose(quaternion_to_rotation_matrix(quat), 1, 2)  # [num_env, 3, 3]
    adj_matrix = torch.zeros(p.shape[0], 6, 6, device=quat.device)
    adj_matrix[:, :3, :3] = rotation_matrix_transpose
    adj_matrix[:, 3:6, 3:6] = rotation_matrix_transpose
    V = skew_symmetric_matrix(p)  # [num_env, 3, 3]
    adj_matrix[:, :3, 3:6] = -torch.bmm(rotation_matrix_transpose, V)  # Batch matrix multiplication
    return adj_matrix

def cart2euler(cart):
    # euler = torch.zeros_like(cart)
    # r = torch.norm(cart, dim=-1)
    # euler[:, 1] = torch.asin(cart[:, 2] / r)
    # euler[:, 2] = torch.atan2(cart[:, 1], cart[:, 0])
        # Initialize Euler angles tensor
    euler = torch.zeros_like(cart)
    
    # Compute the radius (magnitude) of each vector in cart
    r = torch.norm(cart, dim=-1)
    
    # Pitch angle (φ): angle between the vector and its projection on the xy-plane
    # Calculated as the angle between the vector and the z-axis
    # euler[:, 1] = torch.atan2(torch.sqrt(cart[:, 0]**2 + cart[:, 1]**2), cart[:, 2])
    euler[:, 1] = -torch.atan2(cart[:, 2], torch.sqrt(cart[:, 0]**2 + cart[:, 1]**2))
    
    # Yaw angle (θ): angle between the projection of the vector on the xy-plane and the global x-axis
    euler[:, 2] = torch.atan2(cart[:, 1], cart[:, 0])
    # euler[:, 0] = torch.atan2(cart[:, 1], cart[:, 0])
    
    
    # Roll angle (ψ) is not defined in this setup and remains zero
    # euler[:, 0] remains 0 as roll is not applicable for aligning x-axis with the vector and y-axis in the xy-plane
    
    return euler

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
    
def quat_power(quat, t):
    '''
    quat^t = quat_power(quat, t)
    '''
    # quat = [num_envs, 4], t = [num_envs]
    x, y, z, w = torch.unbind(quat, dim=-1)
    angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0).unsqueeze(-1))

    axis_mag = torch.sqrt(x**2 + y**2 + z**2).unsqueeze(-1)
    axis_mag = torch.clamp(axis_mag, min=1e-8)  # Prevent division by zero or near-zero
    axis = torch.stack([x, y, z], dim=-1) / axis_mag
    
    angle_t = angle * t
    w_t = torch.cos(angle_t / 2)
    axis_t = torch.sin(angle_t / 2) * axis
    
    return torch.cat([axis_t, w_t], dim=-1)

def euler_from_matrix(rotation_matrix):
    """
    Convert rotation matrices to Euler angles (XYZ order).

    Args:
        rotation_matrix (torch.Tensor): Rotation matrices. Shape: [num_envs, 3, 3].

    Returns:
        torch.Tensor: Euler angles in radians. Shape: [num_envs, 3].
    """
    # Extract individual elements from the rotation matrix
    m00 = rotation_matrix[:, 0, 0]
    m01 = rotation_matrix[:, 0, 1]
    m02 = rotation_matrix[:, 0, 2]
    m10 = rotation_matrix[:, 1, 0]
    m11 = rotation_matrix[:, 1, 1]
    m12 = rotation_matrix[:, 1, 2]
    m20 = rotation_matrix[:, 2, 0]
    m21 = rotation_matrix[:, 2, 1]
    m22 = rotation_matrix[:, 2, 2]

    # Calculate Euler angles (XYZ order)
    # Calculate pitch (y-axis rotation)
    pitch = torch.atan2(-m20, torch.sqrt(m21**2 + m22**2))

    # Calculate yaw (z-axis rotation)
    yaw = torch.atan2(m10, m00)

    # Calculate roll (x-axis rotation)
    roll = torch.atan2(m21, m22)

    return torch.stack([roll, pitch, yaw], dim=-1)

def rotation_matrix_z(angle):
    """
    Create rotation matrices around the z-axis.

    Args:
        angle (torch.Tensor): Rotation angles in radians. Shape: [num_envs].

    Returns:
        torch.Tensor: Rotation matrices. Shape: [num_envs, 3, 3].
    """
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    
    # Create rotation matrices
    R_z = torch.stack([
        cos_theta, -sin_theta, torch.zeros_like(angle),
        sin_theta, cos_theta, torch.zeros_like(angle),
        torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)
    ], dim=-1).reshape(-1, 3, 3)
    
    return R_z