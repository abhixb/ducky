"""Velocity-tracking walking environment for the Open Duck Mini v2.

A thin, framework-agnostic MuJoCo environment that exposes a Gymnasium-style
API (reset -> obs, info; step -> obs, reward, terminated, truncated, info) but
deliberately avoids subclassing gym.Env so it plugs straight into the custom
PPO. The policy controls the ten leg joints as position targets offset from the
home crouch; the head and antennas are held at their neutral pose.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

from .rewards import (
    tracking_lin_vel, tracking_ang_vel, lin_vel_z, ang_vel_xy, orientation,
    base_height, torque_cost, action_rate, dof_vel, joint_limit, foot_slip,
    feet_air_time,
)
from .randomization import RandomizationConfig, DomainRandomizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = str(REPO_ROOT / "robots/open_duck_mini_v2/scene.xml")

LEG_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll",
               "left_antenna", "right_antenna"]
FEET = ["left_foot", "right_foot"]
TOUCH_SENSORS = ["left_foot_touch", "right_foot_touch"]


@dataclass
class DuckEnvConfig:
    scene: str = DEFAULT_SCENE
    control_dt: float = 0.02            # 50 Hz control
    episode_seconds: float = 20.0
    action_scale: float = 0.25

    # Command sampling ranges (vx, vy in m/s, wz in rad/s).
    cmd_x_range: tuple = (-0.3, 0.6)
    cmd_y_range: tuple = (-0.3, 0.3)
    cmd_yaw_range: tuple = (-1.0, 1.0)
    zero_cmd_prob: float = 0.1

    # Reset noise.
    qpos_noise: float = 0.05
    qvel_noise: float = 0.05

    # Termination.
    min_base_height: float = 0.10
    max_tilt: float = 0.7              # 1 - proj_gravity_z above this terminates

    # Observation noise (sensor model).
    add_noise: bool = True
    noise_gyro: float = 0.2
    noise_gravity: float = 0.05
    noise_dof_pos: float = 0.01
    noise_dof_vel: float = 1.0

    # Gait clock.
    gait_freq: float = 1.5

    reward_weights: dict = field(default_factory=lambda: {
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 0.8,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -1.0,
        "base_height": -10.0,
        "torque": -1e-3,
        "action_rate": -0.01,
        "dof_vel": -1e-4,
        "joint_limit": -1.0,
        "foot_slip": -0.1,
        "feet_air_time": 1.0,
        "alive": 0.5,
    })
    tracking_sigma: float = 0.25
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)


class DuckWalkEnv:
    def __init__(self, config: DuckEnvConfig = None, seed: int = 0):
        self.cfg = config or DuckEnvConfig()
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(self.cfg.scene)
        self.data = mujoco.MjData(self.model)
        self.physics_dt = self.model.opt.timestep
        self.decimation = max(1, round(self.cfg.control_dt / self.physics_dt))
        self.max_steps = int(self.cfg.episode_seconds / self.cfg.control_dt)

        self._cache_indices()
        self._read_home_pose()
        self.randomizer = DomainRandomizer(self.cfg.randomization)

        self.action_dim = len(self.leg_act)
        self.command = np.zeros(3)
        self.last_action = np.zeros(self.action_dim)
        self.feet_air_time = np.zeros(len(FEET))
        self.last_contacts = np.zeros(len(FEET), dtype=bool)
        self.phase = 0.0
        self.step_count = 0

        self.obs_dim = self._obs_template_size()

    # ------------------------------------------------------------------ setup
    def _id(self, objtype, name):
        return mujoco.mj_name2id(self.model, objtype, name)

    def _cache_indices(self):
        m = self.model
        self.leg_qadr = np.array(
            [m.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LEG_JOINTS])
        self.leg_dadr = np.array(
            [m.jnt_dofadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in LEG_JOINTS])
        self.leg_act = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in LEG_JOINTS])
        self.head_act = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in HEAD_JOINTS])

        self.leg_range = m.jnt_range[
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, j) for j in LEG_JOINTS]]

        base_jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        self.base_qadr = m.jnt_qposadr[base_jid]
        self.base_dadr = m.jnt_dofadr[base_jid]

        self.foot_bid = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_BODY, f) for f in FEET])

        self._sensor = {}
        for name in ["orientation", "gyro", "local_linvel"] + TOUCH_SENSORS:
            sid = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
            self._sensor[name] = slice(adr, adr + dim)

    def _read_home_pose(self):
        key = self._id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self.home_qpos = self.model.key_qpos[key].copy()
        self.home_ctrl = self.model.key_ctrl[key].copy()
        self.default_leg_pos = self.home_qpos[self.leg_qadr].copy()
        self.target_height = self.home_qpos[self.base_qadr + 2]

    def _obs_template_size(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data,
                                    self._id(mujoco.mjtObj.mjOBJ_KEY, "home"))
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs().size

    # ------------------------------------------------------------------ helpers
    def _sensordata(self, name):
        return self.data.sensordata[self._sensor[name]]

    @staticmethod
    def _quat_rotate_inverse(q, v):
        w, x, y, z = q
        qvec = np.array([x, y, z])
        return (v * (2.0 * w * w - 1.0)
                - np.cross(qvec, v) * 2.0 * w
                + qvec * (np.dot(qvec, v) * 2.0))

    def _projected_gravity(self):
        quat = self._sensordata("orientation")
        return self._quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

    def _foot_contacts(self):
        return np.array([self._sensordata(s)[0] for s in TOUCH_SENSORS]) > 1.0

    def _foot_velocities_xy(self):
        vels = np.zeros((len(FEET), 2))
        res = np.zeros(6)
        for i, bid in enumerate(self.foot_bid):
            mujoco.mj_objectVelocity(self.model, self.data,
                                     mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
            vels[i] = res[3:5]
        return vels

    def _sample_command(self):
        if self.rng.random() < self.cfg.zero_cmd_prob:
            return np.zeros(3)
        return np.array([
            self.rng.uniform(*self.cfg.cmd_x_range),
            self.rng.uniform(*self.cfg.cmd_y_range),
            self.rng.uniform(*self.cfg.cmd_yaw_range),
        ])

    # ------------------------------------------------------------------ obs
    def _get_obs(self):
        gyro = self._sensordata("gyro").copy()
        gravity = self._projected_gravity()
        dof_pos = self.data.qpos[self.leg_qadr] - self.default_leg_pos
        dof_vel_ = self.data.qvel[self.leg_dadr]

        if self.cfg.add_noise:
            gyro += self.rng.normal(0, self.cfg.noise_gyro, gyro.shape)
            gravity += self.rng.normal(0, self.cfg.noise_gravity, gravity.shape)
            dof_pos += self.rng.normal(0, self.cfg.noise_dof_pos, dof_pos.shape)
            dof_vel_ = dof_vel_ + self.rng.normal(0, self.cfg.noise_dof_vel, dof_vel_.shape)

        clock = np.array([np.sin(self.phase), np.cos(self.phase)])
        return np.concatenate([
            gravity, gyro, dof_pos, dof_vel_, self.last_action, self.command, clock,
        ]).astype(np.float32)

    # ------------------------------------------------------------------ api
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.randomizer.randomize(self.model, self.rng)
        mujoco.mj_resetDataKeyframe(self.model, self.data,
                                    self._id(mujoco.mjtObj.mjOBJ_KEY, "home"))
        n = self.cfg.qpos_noise
        self.data.qpos[self.leg_qadr] += self.rng.uniform(-n, n, self.action_dim)
        self.data.qvel[self.leg_dadr] += self.rng.uniform(
            -self.cfg.qvel_noise, self.cfg.qvel_noise, self.action_dim)
        mujoco.mj_forward(self.model, self.data)

        self.command = self._sample_command()
        self.last_action = np.zeros(self.action_dim)
        self.feet_air_time = np.zeros(len(FEET))
        self.last_contacts = self._foot_contacts()
        self.phase = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = self.default_leg_pos + action * self.cfg.action_scale
        target = np.clip(target, self.leg_range[:, 0], self.leg_range[:, 1])

        self.data.ctrl[self.leg_act] = target
        self.data.ctrl[self.head_act] = self.home_ctrl[self.head_act]

        self.randomizer.maybe_push(self.data, self.base_dadr, self.step_count,
                                   self.cfg.control_dt, self.rng)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self.phase = (self.phase + 2 * np.pi * self.cfg.gait_freq * self.cfg.control_dt) % (2 * np.pi)
        self.step_count += 1

        reward, terms = self._compute_reward(action)
        terminated = self._terminated()
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= 1.0

        self.last_action = action.copy()
        self.last_contacts = self._foot_contacts()
        info = {"reward_terms": terms, "command": self.command.copy()}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ reward
    def _compute_reward(self, action):
        w = self.cfg.reward_weights
        lin_vel = self._sensordata("local_linvel")
        ang_vel = self._sensordata("gyro")
        gravity = self._projected_gravity()
        contacts = self._foot_contacts()
        first_contact = contacts & ~self.last_contacts

        self.feet_air_time += self.cfg.control_dt
        torques = self.data.actuator_force[self.leg_act]
        cmd_norm = np.linalg.norm(self.command)

        terms = {
            "tracking_lin_vel": tracking_lin_vel(self.command[:2], lin_vel[:2], self.cfg.tracking_sigma),
            "tracking_ang_vel": tracking_ang_vel(self.command[2], ang_vel[2], self.cfg.tracking_sigma),
            "lin_vel_z": lin_vel_z(lin_vel[2]),
            "ang_vel_xy": ang_vel_xy(ang_vel),
            "orientation": orientation(gravity),
            "base_height": base_height(self.data.qpos[self.base_qadr + 2], self.target_height),
            "torque": torque_cost(torques),
            "action_rate": action_rate(action, self.last_action),
            "dof_vel": dof_vel(self.data.qvel[self.leg_dadr]),
            "joint_limit": joint_limit(self.data.qpos[self.leg_qadr],
                                       self.leg_range[:, 0], self.leg_range[:, 1]),
            "foot_slip": foot_slip(self._foot_velocities_xy(), contacts),
            "feet_air_time": feet_air_time(self.feet_air_time, first_contact, cmd_norm),
            "alive": 1.0,
        }
        self.feet_air_time[contacts] = 0.0

        total = sum(w[k] * terms[k] for k in terms) * self.cfg.control_dt
        return float(total), terms

    def _terminated(self):
        z = self.data.qpos[self.base_qadr + 2]
        upright = -self._projected_gravity()[2]  # 1.0 when perfectly upright
        if z < self.cfg.min_base_height:
            return True
        if (1.0 - upright) > self.cfg.max_tilt:
            return True
        if not np.all(np.isfinite(self.data.qpos)):
            return True
        return False
