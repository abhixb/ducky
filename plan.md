# Open Duck Mini v2 - RL Project Plan

This document captures the full plan from the start of the session, including the
decisions made via clarifying questions, work already completed, and the remaining
implementation plan.

---

## 0. Project context

- Project: Open Duck Mini v2 - an open-source ~42cm bipedal robot, a miniature clone
  of the Disney BD-X droid. Goal here: train our own walking policy from scratch.
- The original repo (`apirrone/mini_BDX`, branch `v2`) was a "hub" repo with many
  undocumented scripts. We stripped it down to just the robot description and are
  rewriting all scripts and environments ourselves.
- Upstream moved RL training to MuJoCo Playground (`Open_Duck_Playground`); we are
  deliberately building our own training stack instead.

---

## 1. Decisions made (clarifying questions and answers)

### Q1 - Dependency install target
- Question: Where to install dependencies?
- Answer: New conda env `ducky` (Python 3.11). Chosen because the pinned package
  versions (gymnasium 0.29.1, sb3 2.3.2, imitation 1.0.0, mujoco 3.1.5) are most
  compatible with Python 3.10/3.11, not the system's 3.12.

### Q2 - Dependency set
- Question: Which dependency set?
- Answer: `all` (sim + training). Not the `robot` extra (no real-hardware deps like
  pypot for now).

### Q3 - Which robot model(s) to keep
- Question: Keep which robot models under `mini_bdx/robots/`?
- Answer: Both `open_duck_mini_v2` (the duck, primary) and `bdx` (legacy v1, kept for
  reference). Working focus is `open_duck_mini_v2`.

### Q4 - Keep pretrained ONNX policies?
- Question: Keep `BEST_WALK_ONNX*.onnx`?
- Answer: Delete them. Full clean slate; we train our own.

### Q5 - Keep repo scaffolding?
- Question: Keep LICENSE / README / .gitignore / git history?
- Answer: Keep LICENSE, .gitignore, and git history. Delete README/docs/experiments/
  print/python package.

### Q6 - First build step
- Question: What to build first?
- Answer: MJCF cleanup first (proper RL model: floating base + position actuators +
  BAM-identified params + foot contact sensors), then build the env on top.

### Q7 - RL framework
- Question: Which RL framework?
- Answer: Custom PyTorch PPO (write our own training loop for full control).

---

## 2. Work already completed

### 2.1 Environment
- Created conda env `ducky` (Python 3.11).
- Installed via pip: mujoco==3.1.5, mujoco-python-viewer==0.1.4,
  onshape-to-robot==0.3.25, gymnasium[mujoco]==0.29.1, stable-baselines3[extra]==2.3.2,
  sb3_contrib==2.3.0, FramesViewer, inputs==0.5, imitation==1.0.0, h5py==3.11.0.
- Installed `placo` via conda-forge (pip build from source failed: pinocchio needs
  Boost). conda binary works.
- Installed `onnxruntime` (for loading/running policies if needed later).
- Activate with: `conda activate ducky`.

### 2.2 Repo cleanup
- Deleted (recoverable via git history): docs/, experiments/, print/, both .onnx
  policies, README.md, thanks.md, FUNDING.yml, setup.cfg, pyproject.toml, and the
  `mini_bdx/mini_bdx/` python package (+ egg-info, pycache).
- Kept: `mini_bdx/robots/open_duck_mini_v2/`, `mini_bdx/robots/bdx/`, LICENSE,
  .gitignore, .git.
- Verified all three MuJoCo scenes still load.
- NOTE: deletions are staged but NOT committed.

---

## 3. Robot model reference (open_duck_mini_v2)

### 3.1 Actuated joints (16 total)

Legs (5 per leg, 10 total) - the ones that matter for walking:

| Left            | Right            | Range (rad)            |
|-----------------|------------------|------------------------|
| left_hip_yaw    | right_hip_yaw    | +/-0.52                |
| left_hip_roll   | right_hip_roll   | +/-0.44                |
| left_hip_pitch  | right_hip_pitch  | -1.22..0.52 (mirrored) |
| left_knee       | right_knee       | +/-1.57                |
| left_ankle      | right_ankle      | +/-1.57                |

Head/neck (4): neck_pitch, head_pitch, head_yaw, head_roll
Antennas (2, cosmetic): left_antenna, right_antenna

Joint qpos order (floating base model): freejoint(7) then
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll, left_antenna, right_antenna,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle.
(Note the interleaved ordering: left leg, then head/antennas, then right leg.)

### 3.2 Existing model files

| File              | includes        | Base               | Actuators            | Use            |
|-------------------|-----------------|--------------------|----------------------|----------------|
| scene.xml         | robot_motors.xml| floating (z=0.17)  | torque (`<motor>`)   | floating sim   |
| scene_position.xml| robot.xml       | fixed              | position PD (kp=9.5) | posing         |
| robot.xml         | -               | fixed (freejoint commented) | position PD | kinematic tree |
| robot_motors.xml  | -               | floating           | torque               | torque tree    |

### 3.3 Key physical facts (measured)
- Floating model: nq=23 (7 base + 16 joints), nu=16.
- At zero joint angles, feet sit ~0.194 below the base origin; standing base height
  ~0.195 for straight-leg ground contact (real init pose will use bent knees).
- BAM hardware-identified actuator params (in a comment in robot.xml): damping=1.44,
  frictionloss=0.19, armature=0.012, kp=19.6, kv=0, forcerange=+/-3.69. These are the
  sim2real-grade values for the Feetech STS3215 leg servos. Use these, not the active
  defaults (damping=1.0, kp=9.5).
- IMU body `bno055` on the trunk; `left_foot` / `right_foot` frame bodies at the feet.

---

## 4. Plan: Phase 1 - MJCF cleanup (build the RL model)

Goal: a single clean RL scene with a floating base, position/PD actuators using
BAM params, a proper visual/collision split, an IMU site, and foot contact sensors.

### 4.1 Edits to `robot.xml`
1. Default block:
   - Make geoms visual-only by default: `<geom contype="0" conaffinity="0" group="2"/>`.
   - Add a `collision` subclass: `<geom contype="1" conaffinity="1" group="3"/>`.
   - Set BAM joint defaults: `<joint damping="1.44" frictionloss="0.19" armature="0.012"/>`.
   - Set BAM position defaults: `<position kp="19.6" kv="0.0" forcerange="-3.69 3.69"/>`.
2. Base body:
   - Enable floating base: replace commented freejoint with `<freejoint name="floating_base"/>`.
   - Set base pos z to ~0.195.
   - Add IMU site in trunk_assembly: `<site name="imu" pos="-0.024 0 0.088" size="0.005"/>`.
3. Feet: add `class="collision"` to the foot geoms (foot_side, foot_bottom_tpu,
   foot_bottom_pla, foot_top) on both feet, reusing the foot meshes as convex-hull
   collision against the floor. Everything else stays visual-only (fast, stable).
4. Add foot touch sites (box) at each foot for contact sensing.
5. Add a `<sensor>` block:
   - framequat (orientation), gyro, accelerometer, velocimeter on `imu` site.
   - framepos for base position.
   - touch sensors `left_foot_touch` / `right_foot_touch`.
6. Add a `home` keyframe with a stable standing crouch pose (computed/tuned below).

### 4.2 Edits to `scene.xml`
- Change include from `robot_motors.xml` to `robot.xml` (use the cleaned PD model).
- Add `<option timestep="0.005"/>` (200 Hz physics; control decimation handled in env,
  target 50 Hz control).
- Keep floor, lighting, friction default (1.5 0.01 0.0006).

### 4.3 Verification (must pass before Phase 2)
- Model loads; nq=23, nu=16, sensors present.
- Drop/settle test under PD holding the home pose: robot stands stably without
  exploding or sinking through the floor.
- Foot touch sensors read > 0 when standing (resize touch sites if they read 0).
- Determine and bake a stable standing crouch keyframe (approx start point:
  hip_pitch +/-0.5, knee +/-1.0, ankle +/-0.5 with mirrored signs; verify flat feet).

---

## 5. Plan: Phase 2 - Walking environment (from scratch)

A clean MuJoCo-backed env (Gymnasium-style API, but framework-agnostic so it plugs
into our custom PPO).

### 5.1 Control
- Action space: 16 target joint positions (or just the 10 leg joints with head/antennas
  held at neutral for v1). Actions are deltas around the default pose, scaled, clipped
  to joint ranges, fed to the position actuators.
- Control at 50 Hz; physics at 200 Hz (decimation 4).

### 5.2 Observation
- Base orientation (gravity vector / projected gravity from IMU quat).
- Base angular velocity (gyro).
- Joint positions and velocities (relative to default pose).
- Previous action.
- Velocity command (vx, vy, wz).
- Gait phase clock (sin/cos) if using a periodic gait prior.
- Optionally a short history stack of observations.

### 5.3 Reward (velocity-tracking locomotion)
- Linear velocity tracking (xy) and angular velocity tracking (yaw).
- Penalties: orientation/upright, base height, torque/energy, action rate,
  joint limits, foot slip, undesired contacts.
- Optional foot air-time / contact-schedule reward to encourage stepping.

### 5.4 Termination
- Fall detection (base orientation past threshold or base height too low).
- Episode timeout.

### 5.5 Domain randomization (for sim2real)
- Friction, mass, IMU noise, action latency, motor strength scale, push perturbations.

---

## 6. Plan: Phase 3 - Custom PyTorch PPO

- Vectorized rollout over N parallel MuJoCo envs (start with Python multiprocessing /
  simple vec wrapper; consider mujoco MJX/JAX later if throughput is the bottleneck).
- Actor-critic MLP (separate or shared trunk), tanh-squashed Gaussian policy.
- PPO core: GAE(lambda), clipped surrogate objective, value clipping, entropy bonus,
  advantage normalization, gradient clipping, LR schedule.
- Observation normalization (running mean/std).
- Logging (tensorboard), checkpointing, eval rollouts with the mujoco viewer.
- Export trained policy (torch -> ONNX) for the onboard runtime later.

---

## 7. Suggested project layout

```
open_duck_mini_v2/        # robot description (current robots/open_duck_mini_v2)
  robot.xml               # cleaned RL model (floating + PD + BAM + sensors)
  scene.xml               # RL scene (includes robot.xml + floor + options)
  *.stl                   # meshes
env/
  duck_env.py             # walking environment
  rewards.py              # reward terms
  randomization.py        # domain randomization
rl/
  ppo.py                  # custom PPO trainer
  networks.py             # actor-critic
  rollout.py              # vectorized rollout buffer
scripts/
  view.py                 # load + visualize + poke joints
  train.py                # training entrypoint
  play.py                 # run a checkpoint in the viewer
  export_onnx.py          # export policy
```

---

## 8. Immediate next steps
1. Apply the `robot.xml` / `scene.xml` MJCF cleanup edits (Section 4).
2. Run the verification + tune and bake the standing keyframe.
3. Build a minimal `scripts/view.py` to sanity-check the model visually.
4. Implement the walking env (Section 5).
5. Implement custom PPO (Section 6) and start training.

## Notes / open items
- Decide whether v1 policy controls all 16 joints or only the 10 leg joints (head and
  antennas held neutral). Recommendation: start with 10 leg joints for a simpler,
  faster-to-train walker, then add head control.
- The deletions in the original repo are staged but not committed; commit when ready.
- conda env `ducky` already has mujoco/torch/etc. installed and is ready to use.
