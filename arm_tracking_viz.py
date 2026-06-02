"""
Visual companion to the numeric arm-tracking test in diagnostic_tidybot_push_easy.py.

Drives the TidyBot arm (via the env's delta control) through a sequence of NAMED
target joint poses with the base held fixed, so you can *see* the arm reach each
commanded pose and watch the joint error shrink. This is the visual proof that
"the arm goes where you think it should go" — it is NOT the push task (the base
does not move and the cube is ignored).

Outputs (in the repo root):
  arm_tracking.html  - interactive brax viewer (open on a machine WITH WebGL, e.g.
                       your laptop — not the headless box). No text overlay; use the
                       printed timeline log + the frame slider to know what's commanded.
  arm_tracking.mp4   - fixed-camera video with a burned-in caption showing the current
                       target pose and the live max joint error (osmesa software render;
                       plays anywhere, no WebGL/internet needed).

A timeline log (frame range -> target -> achieved error) is printed at the end.

Usage: uv run arm_tracking_viz.py [--cpu]
"""

import sys
import os

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
# Offscreen software GL for the MuJoCo renderer (EGL fails on this box; osmesa works).
# Must be set before mujoco is imported (the env import below pulls it in).
os.environ.setdefault("MUJOCO_GL", "osmesa")

import jax
import jax.numpy as jnp
import numpy as np

from brax.io import html
import mujoco
import mediapy
from PIL import Image, ImageDraw

from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy

print("Initializing TidyBotPushEasy...", flush=True)
env = TidyBotPushEasy(backend="mjx")
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

# Delta-control inverse mapping: env moves target by action*0.25*multiplier per step.
arm_min = np.array(env.arm_joint_range[:, 0])
arm_max = np.array(env.arm_joint_range[:, 1])
arm_mult = (arm_max - arm_min) / 2.0


def arm_command(q_tgt, arm_q):
    a = np.zeros(11)
    a[3:10] = np.clip((q_tgt - arm_q) / (0.25 * arm_mult), -1.0, 1.0)
    a[10] = -1.0  # gripper open (irrelevant here; base dims 0:2 stay 0 -> base holds)
    return jnp.array(a)


# Named target poses (all within joint limits). The arm should visibly reach each.
SEQUENCE = [
    ("ARM_DEFAULT", np.array([-np.pi / 2, 1.4, np.pi, -1.2, 0.0, 0.5, np.pi / 2])),
    ("ARM_LOWER",   np.array([-np.pi / 2, 1.9, np.pi, -1.0, 0.0, 0.5, np.pi / 2])),
    ("interior",    np.array([0.3, 1.0, 0.5, -1.5, -0.5, 0.8, 0.6])),
    ("twist_j1",    np.array([1.5, 1.0, 0.0, -1.0, 0.0, 0.5, 0.0])),
    ("ARM_DEFAULT", np.array([-np.pi / 2, 1.4, np.pi, -1.2, 0.0, 0.5, np.pi / 2])),
]
HOLD = 100  # steps held on each target (reach + settle)

rng = jax.random.PRNGKey(0)
state = reset_fn(rng)

pipeline_states = [state.pipeline_state]
captions = ["reset (start)"]
log = []

frame = 1
for name, q_tgt in SEQUENCE:
    seg_start = frame
    for _ in range(HOLD):
        arm_q = np.array(state.obs[3:10])
        state = step_fn(state, arm_command(q_tgt, arm_q))
        err = float(np.max(np.abs(np.array(state.obs[3:10]) - q_tgt)))
        pipeline_states.append(state.pipeline_state)
        captions.append(f"target: {name}   max joint err: {err:.3f} rad")
        frame += 1
    final_err = np.abs(np.array(state.obs[3:10]) - q_tgt)
    log.append((name, seg_start, frame - 1, float(final_err.max()), int(final_err.argmax())))

print("\n=== arm-tracking timeline (base held; arm driven to each target) ===")
for name, a, b, mx, j in log:
    print(f"  frames {a:4d}-{b:4d}   target={name:12s}   final max|q-q_tgt|={mx:.3f} rad "
          f"(worst joint arm_{j + 1})")

# --- HTML (interactive; needs WebGL) ---
with open("arm_tracking.html", "w") as f:
    f.write(html.render(env.sys, pipeline_states))
print(f"\nSaved arm_tracking.html  ({len(pipeline_states)} frames)")

# --- MP4 with burned-in caption (plays anywhere) ---
mj_model = env.sys.mj_model
mj_data = mujoco.MjData(mj_model)

cam = mujoco.MjvCamera()  # fixed camera framed on the (stationary) robot
cam.lookat[:] = [0.0, 0.15, 0.45]
cam.distance = 2.3
cam.azimuth = 110.0
cam.elevation = -15.0

renderer = mujoco.Renderer(mj_model, height=480, width=640)
frames = []
for ps, cap in zip(pipeline_states, captions):
    mj_data.qpos[:] = np.array(ps.q)
    mj_data.qvel[:] = np.array(ps.qd)
    mujoco.mj_forward(mj_model, mj_data)
    renderer.update_scene(mj_data, camera=cam)
    img = Image.fromarray(renderer.render())
    ImageDraw.Draw(img).text((10, 10), cap, fill=(255, 255, 0))
    frames.append(np.array(img))
renderer.close()

mediapy.write_video("arm_tracking.mp4", frames, fps=20)
print(f"Saved arm_tracking.mp4  ({len(frames)} frames @ 20 fps)")
