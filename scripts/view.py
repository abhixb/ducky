"""Load the Open Duck Mini v2 scene in the MuJoCo viewer to sanity-check the
model: holds the home keyframe under PD, with an optional joint sweep to verify
the actuators and joint directions visually.

  python scripts/view.py                # hold the home crouch
  python scripts/view.py --sweep        # gently sweep the leg joints
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = str(REPO_ROOT / "robots/open_duck_mini_v2/scene.xml")
LEG_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--sweep", action="store_true",
                        help="sinusoidally sweep the leg joints around the home pose")
    parser.add_argument("--amp", type=float, default=0.2, help="sweep amplitude (rad)")
    parser.add_argument("--freq", type=float, default=0.5, help="sweep frequency (Hz)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)

    base_ctrl = data.ctrl.copy()
    leg_act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in LEG_JOINTS]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()
            data.ctrl[:] = base_ctrl
            if args.sweep:
                phase = 2 * np.pi * args.freq * (time.time() - start)
                data.ctrl[leg_act] = base_ctrl[leg_act] + args.amp * np.sin(phase)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
