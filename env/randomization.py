"""Domain randomization for sim2real transfer.

The randomizer perturbs physical properties of a per-environment MjModel at
reset (friction, link masses, motor strength) and injects disturbances at
runtime (random base pushes). Sensor/observation noise is handled separately
by the environment so the policy's input model stays in one place.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomizationConfig:
    enabled: bool = True
    friction_range: tuple = (0.6, 1.4)        # multiplies nominal sliding friction
    mass_scale_range: tuple = (0.9, 1.1)      # multiplies each link mass/inertia
    motor_strength_range: tuple = (0.9, 1.1)  # multiplies actuator kp
    push_interval_s: float = 2.0
    push_vel: float = 0.4                      # m/s impulse added to the base


class DomainRandomizer:
    def __init__(self, config: RandomizationConfig):
        self.cfg = config
        self._nominal = None

    def _snapshot(self, model):
        self._nominal = {
            "geom_friction": model.geom_friction.copy(),
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "gainprm": model.actuator_gainprm.copy(),
            "biasprm": model.actuator_biasprm.copy(),
        }

    def randomize(self, model, rng):
        """Resample physical parameters from the nominal model into `model`."""
        if not self.cfg.enabled:
            return
        if self._nominal is None:
            self._snapshot(model)
        nom = self._nominal

        f = rng.uniform(*self.cfg.friction_range)
        model.geom_friction[:, 0] = nom["geom_friction"][:, 0] * f

        m = rng.uniform(*self.cfg.mass_scale_range, size=model.nbody)
        model.body_mass[:] = nom["body_mass"] * m
        model.body_inertia[:] = nom["body_inertia"] * m[:, None]

        s = rng.uniform(*self.cfg.motor_strength_range, size=model.nu)
        model.actuator_gainprm[:, 0] = nom["gainprm"][:, 0] * s
        model.actuator_biasprm[:, 1] = nom["biasprm"][:, 1] * s

    def maybe_push(self, data, base_dof_adr, step, control_dt, rng):
        """Apply a random horizontal velocity kick to the base on an interval."""
        if not self.cfg.enabled or self.cfg.push_interval_s <= 0:
            return
        interval = max(1, int(self.cfg.push_interval_s / control_dt))
        if step > 0 and step % interval == 0:
            kick = rng.uniform(-self.cfg.push_vel, self.cfg.push_vel, size=2)
            data.qvel[base_dof_adr:base_dof_adr + 2] += kick
