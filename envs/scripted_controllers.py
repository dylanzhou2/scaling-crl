"""Hand-coded skill controllers for scripted-ball composition.

Each controller returns an 11-dim env-space action (the Mahalanobis-ball CENTER)
given the shared 20-dim TidyBot state and a synthesized goal. Batched (leading
dims handled via `...`). They are deliberately simple — they define the ball
center; the composition RL learns bounded corrections *within* the ball, so the
controllers need only be "drive-like" / "push-like," not optimal.

Action layout (11): [base_x, base_y, base_th (0:3), arm_j1..j7 (3:10), gripper (10)].

  base_control: holonomic P-controller to a target xy. Heading held (base_th=0),
                arm held (zero deltas → tucked pose persists under delta control).
  push_control: drive the arm toward a contact pose behind the cube and sweep it
                to push the cube toward a target x. Gripper held closed. Approximate
                — a tuning target for the real runs.
"""
import jax.numpy as jnp

# Arm pose that puts the fingertips just behind the cube at contact height
# (shared with envs/mobile_manipulation/tidybot_push_aside.py:ARM_Q_CONTACT).
ARM_Q_CONTACT = jnp.array([-1.71, 1.405, 2.98, -1.237, 0.198, -0.393, 1.351])


def base_control(state, goal_xy, base_scale: float = 0.2):
    """Drive the holonomic base toward goal_xy; hold heading; hold the arm."""
    base_xy = state[..., 0:2]
    drive = jnp.clip((goal_xy - base_xy) / base_scale, -1.0, 1.0)
    a = jnp.zeros(state.shape[:-1] + (11,))
    a = a.at[..., 0:2].set(drive)  # base x/y; a[...,2]=0 heading; arm/gripper held
    return a


def push_control(state, cube_goal_xy, arm_contact=ARM_Q_CONTACT,
                 push_scale: float = 0.2):
    """Move the arm toward the contact pose and sweep j1 to push the cube +x."""
    arm = state[..., 3:10]
    arm_delta = jnp.clip(arm_contact - arm, -1.0, 1.0)        # drive toward contact
    err_x = cube_goal_xy[..., 0] - state[..., 10]             # push along x
    sweep = jnp.clip(err_x / push_scale, -1.0, 1.0)
    a = jnp.zeros(state.shape[:-1] + (11,))
    a = a.at[..., 3:10].set(arm_delta)
    a = a.at[..., 3].add(sweep)                               # extra j1 push sweep
    a = a.at[..., 10].set(-1.0)                               # gripper closed
    return jnp.clip(a, -1.0, 1.0)


# env_id -> controller, used by the collector and the composition trainer.
CONTROLLERS = {
    "tidybot_navigate": base_control,
    "tidybot_push_aside": push_control,
}
