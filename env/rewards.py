"""Pure reward/penalty terms for the walking task.

Each function returns a scalar describing one component of the reward. The
environment owns the weights and the sign convention: tracking terms return a
positive bounded value in [0, 1], penalties return a non-negative magnitude
that the environment negates via its weight. Keeping these stateless makes the
reward shaping easy to test and to tune from the env config alone.
"""

import numpy as np


def tracking_lin_vel(cmd_xy, vel_xy, sigma):
    """Exponential tracking of the commanded planar velocity (body frame)."""
    err = np.sum(np.square(cmd_xy - vel_xy))
    return np.exp(-err / sigma)


def tracking_ang_vel(cmd_yaw, yaw_rate, sigma):
    """Exponential tracking of the commanded yaw rate."""
    err = np.square(cmd_yaw - yaw_rate)
    return np.exp(-err / sigma)


def lin_vel_z(vel_z):
    """Penalise vertical bobbing of the base."""
    return np.square(vel_z)


def ang_vel_xy(ang_vel):
    """Penalise roll/pitch angular velocity of the base."""
    return np.sum(np.square(ang_vel[:2]))


def orientation(proj_gravity):
    """Penalise tilt: the projected gravity x/y vanish when perfectly upright."""
    return np.sum(np.square(proj_gravity[:2]))


def base_height(z, target):
    return np.square(z - target)


def torque_cost(torques):
    return np.sum(np.square(torques))


def action_rate(action, last_action):
    return np.sum(np.square(action - last_action))


def dof_vel(qvel):
    return np.sum(np.square(qvel))


def joint_limit(q, lower, upper, soft_ratio=0.9):
    """Penalise positions that exit a soft band inside the hard joint limits."""
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * soft_ratio
    over = np.clip(np.abs(q - mid) - half, 0.0, None)
    return np.sum(np.square(over))


def foot_slip(foot_vel_xy, contacts):
    """Penalise horizontal foot velocity while that foot is loaded."""
    speed = np.sum(np.square(foot_vel_xy), axis=-1)
    return np.sum(speed * contacts)


def feet_air_time(air_time, first_contact, cmd_norm, target=0.5):
    """Reward steps that keep each foot airborne for ~target seconds.

    Only credited on the control step where a foot first touches down, and only
    when a non-trivial velocity is commanded so the robot is not rewarded for
    marching in place.
    """
    reward = np.sum((air_time - target) * first_contact)
    return reward * (cmd_norm > 0.1)
