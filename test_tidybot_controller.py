"""
TidyBot push controller test — validates env geometry before Phase 3 training.

Two-phase scripted policy:
  Phase 1 (steps 0–150):  navigate base to line up behind cube; lower arm to cube height
  Phase 2 (steps 150–500): push base+arm forward toward goal

If this achieves reward > 0 in at least 1/10 episodes, the env is wired correctly.
If not, the geometry or reward function still has a bug.

Usage: uv run test_tidybot_controller.py [--cpu] [--viz] [--mp4]
  --cpu  force JAX onto CPU
  --viz  also render episode 0 to tidybot_controller_rollout.html (brax.io.html;
         needs a browser with WebGL — won't render on a headless box)
  --mp4  also render episode 0 to tidybot_controller_rollout.mp4 via MuJoCo's
         offscreen software renderer (no GPU/WebGL/display needed)
"""

import sys
import os

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

VIZ = "--viz" in sys.argv
MP4 = "--mp4" in sys.argv

if MP4:
    # Offscreen software GL for the MuJoCo renderer. Must be set before mujoco is
    # imported (the env import below pulls it in). On this box EGL fails to get a
    # context (same as the browser's WebGL); osmesa software rendering works.
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import html

from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy

print("Initializing TidyBotPushEasy...", flush=True)
env = TidyBotPushEasy(backend="mjx")
reset_fn = jax.jit(env.reset)
step_fn  = jax.jit(env.step)

# Arm action mapping. TidyBotEnv uses delta control: each step the env moves a joint
# target by action*0.25*multiplier (clamped), so to drive toward an absolute target
# pose q_tgt we command (q_tgt - current)/(0.25*multiplier). multiplier = half the
# joint range; offsets are ~0 here (Kinova ranges are symmetric about 0).
arm_min  = np.array(env.arm_joint_range[:, 0])
arm_max  = np.array(env.arm_joint_range[:, 1])
arm_mult = (arm_max - arm_min) / 2.0

def arm_delta_action(q_tgt, arm_q):
    return np.clip((q_tgt - arm_q) / (0.25 * arm_mult), -1.0, 1.0)

NUM_EPISODES  = 10
EPISODE_LEN   = 500
PHASE1_STEPS  = 150   # navigate + lower arm
PHASE2_STEPS  = 350   # push

# obs layout: base(0-2), arm(3-9), cube(10-12), eef_pos(13-15), eef_vel(16-18), finger(19), goal(20-22)
BASE  = slice(0, 3)
ARM   = slice(3, 10)
CUBE  = slice(10, 13)
EEF   = slice(13, 16)
GOAL  = slice(20, 23)

# arm_q_default from _get_initial_state — used as push target
ARM_DEFAULT = np.array([-np.pi/2, 1.4, np.pi, -1.2, 0.0, 0.5, np.pi/2])
# Push pose: j2=2.05 drops the wrist to ~cube height; fingers reach ~0.75 m ahead and
# hang down through the cube's z-extent (cube spans z=0.04-0.10), so they can shove it.
ARM_LOWER   = np.array([-np.pi/2, 2.05, np.pi, -1.0, 0.0, 0.5, np.pi/2])

# Finger reach is ~0.75 m, so STANDOFF ~0.85 places the fingers just behind the cube
# in Phase 1; Phase 2 tracks STANDOFF-PUSH_DEPTH=0.70 behind the (moving) cube, pressing
# the fingers into it and driving it toward the goal as the base follows.
STANDOFF   = 0.85
PUSH_DEPTH = 0.15

print(f"\nRunning {NUM_EPISODES} episodes, {EPISODE_LEN} steps each")
print(f"Phase 1 ({PHASE1_STEPS} steps): align base + lower arm")
print(f"Phase 2 ({PHASE2_STEPS} steps): push base+arm toward goal\n")

rewards_per_ep = []
min_dists      = []
max_rewards    = []

rollout_states = []  # pipeline states for episode 0, captured when --viz or --mp4

rng = jax.random.PRNGKey(42)

for ep in range(NUM_EPISODES):
    rng, ep_key = jax.random.split(rng)
    state = reset_fn(ep_key)

    if (VIZ or MP4) and ep == 0:
        rollout_states.append(state.pipeline_state)

    ep_reward = 0.0
    min_dist  = float("inf")

    for t in range(EPISODE_LEN):
        obs      = np.array(state.obs)
        base_pos = obs[BASE]   # (x, y, th)
        arm_q    = obs[ARM]    # 7 joint angles
        cube_pos = obs[CUBE]   # (x, y, z)
        eef_pos  = obs[EEF]    # (x, y, z)
        goal_pos = obs[GOAL]   # (x, y, z)

        eef_cube_dist = float(np.linalg.norm(eef_pos - cube_pos))
        min_dist = min(min_dist, eef_cube_dist)

        cube_to_goal = goal_pos[:2] - cube_pos[:2]
        push_dir     = cube_to_goal / (np.linalg.norm(cube_to_goal) + 1e-6)

        if t < PHASE1_STEPS:
            # ── Phase 1: position base behind cube; lower arm toward cube height ──

            # Target: base sits STANDOFF behind cube so arm EEF (at y=0.545 relative)
            # lands at cube position when base reaches target
            approach_offset = -push_dir * STANDOFF
            target_base_xy  = cube_pos[:2] + approach_offset
            base_err        = target_base_xy - base_pos[:2]
            base_action     = np.clip(base_err / 0.2, -1.0, 1.0)

            # Target heading: align robot with push direction
            # The arm reaches along the base +y axis (world angle = heading + pi/2),
            # so to point the arm along push_dir the heading must be push_angle - pi/2.
            target_th  = float(np.arctan2(push_dir[1], push_dir[0])) - np.pi / 2
            th_err     = target_th - base_pos[2]
            th_err     = (th_err + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            th_action  = np.clip(th_err / 0.2, -1.0, 1.0)

            # Lower arm: interpolate from default to lower pose over phase 1
            alpha    = t / PHASE1_STEPS
            arm_tgt  = (1 - alpha) * ARM_DEFAULT + alpha * ARM_LOWER
            arm_action = arm_delta_action(arm_tgt, arm_q)

        else:
            # ── Phase 2: hold lowered pose; keep base pressing into the cube ──
            # Track a point just *inside* the cube along push_dir so the EEF keeps
            # contact and drives the cube toward the goal. Closed-loop on the moving
            # cube → naturally stops when the cube reaches the goal (no runaway).
            target_base_xy = cube_pos[:2] - push_dir * (STANDOFF - PUSH_DEPTH)
            base_err       = target_base_xy - base_pos[:2]
            base_action    = np.clip(base_err / 0.2, -1.0, 1.0)

            # The arm reaches along the base +y axis (world angle = heading + pi/2),
            # so to point the arm along push_dir the heading must be push_angle - pi/2.
            target_th  = float(np.arctan2(push_dir[1], push_dir[0])) - np.pi / 2
            th_err     = (target_th - base_pos[2] + np.pi) % (2 * np.pi) - np.pi
            th_action  = np.clip(th_err / 0.2, -1.0, 1.0)

            arm_action = arm_delta_action(ARM_LOWER, arm_q)  # hold lowered push pose

        action = np.concatenate([base_action, [th_action], arm_action, [-1.0]])
        rng, step_key = jax.random.split(rng)
        state = step_fn(state, jnp.array(action))

        if (VIZ or MP4) and ep == 0:
            rollout_states.append(state.pipeline_state)

        r = float(state.reward)
        ep_reward = max(ep_reward, r)
        if r > 0:
            print(f"  ep={ep} t={t:3d}  REWARD={r:.2f}  eef-cube={eef_cube_dist:.3f}m  cube-goal={np.linalg.norm(cube_pos-goal_pos):.3f}m")

    rewards_per_ep.append(ep_reward)
    min_dists.append(min_dist)
    print(f"  ep={ep}  max_reward={ep_reward:.2f}  min_eef-cube={min_dist:.4f}m  "
          f"final_cube-goal={np.linalg.norm(np.array(state.obs)[CUBE]-goal_pos):.3f}m")

print(f"\n{'='*60}")
print(f"Summary ({NUM_EPISODES} episodes)")
print(f"{'='*60}")
n_success = sum(r > 0 for r in rewards_per_ep)
print(f"Episodes with reward > 0 : {n_success}/{NUM_EPISODES}")
print(f"Min EEF-cube dist (best) : {min(min_dists):.4f} m")
print(f"Mean min EEF-cube dist   : {np.mean(min_dists):.4f} m")

if n_success > 0:
    print("\n[PASS] Controller achieved reward — env geometry and reward are correct.")
    print("       Phase 3 training should learn.")
elif min(min_dists) < 0.15:
    print("\n[PARTIAL] EEF reached cube but no reward. Check success threshold or goal geometry.")
else:
    print("\n[FAIL] EEF never contacted cube. Arm controller or geometry still misconfigured.")

if VIZ and rollout_states:
    out_path = "tidybot_controller_rollout.html"
    html_string = html.render(env.sys, rollout_states)
    with open(out_path, "w") as f:
        f.write(html_string)
    print(f"\nSaved episode-0 rollout to {out_path}  ({len(rollout_states)} frames)")

if MP4 and rollout_states:
    import mujoco
    import mediapy

    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Explicit camera framed on the floor-level workspace. The default free camera
    # looks at z~0.69 (inflated by the upward arm) from ~3m back, leaving the cube
    # (z=0.07) and goal (z=0.03) tiny. Center on the cube/goal/base centroid instead.
    def _bpos(name):
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.array(mj_data.xpos[bid]) if bid >= 0 else None

    mj_data.qpos[:] = np.array(rollout_states[0].q)
    mujoco.mj_forward(mj_model, mj_data)
    pts = [p for p in (_bpos("cube"), _bpos("goal_marker"), _bpos("base_link")) if p is not None]
    lookat = np.mean(pts, axis=0) if pts else np.array(mj_model.stat.center)
    lookat[2] = 0.25  # keep the floor workspace and the robot body both in frame

    cam = mujoco.MjvCamera()  # type defaults to free
    cam.lookat[:] = lookat
    cam.distance = 2.5
    cam.azimuth = 110.0
    cam.elevation = -20.0

    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    for ps in rollout_states:
        mj_data.qpos[:] = np.array(ps.q)
        mj_data.qvel[:] = np.array(ps.qd)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render())

    renderer.close()  # free the GL context now; avoids a noisy __del__ at shutdown

    fps = round(1.0 / float(env.dt)) if float(env.dt) > 0 else 30
    out_mp4 = "tidybot_controller_rollout.mp4"
    mediapy.write_video(out_mp4, frames, fps=fps)
    print(f"\nSaved episode-0 rollout video to {out_mp4}  ({len(frames)} frames @ {fps} fps)")
