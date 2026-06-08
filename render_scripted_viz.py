"""Render HTML visualizations of the scripted skills and the hallway task.

Produces brax-viewer HTML for:
  * the hallway TASK   — a scripted navigate -> push -> navigate scaffold, so you
                         can see the corridor, the blocking cube, and the robot;
  * the BASE skill     — the navigate-env rollouts used to gather base-ball data;
  * the PUSH skill      — the push_aside-env rollouts (arm-start noise, as in the
                         collection) used to gather push-ball data.

  uv run render_scripted_viz.py        # writes viz_*.html in the cwd
"""
import jax
import jax.numpy as jnp
from brax.io import html

from envs.scripted_controllers import base_control, push_control
from train_residual_mab import make_env

SD, STANDOFF, PUSH_TARGET_X, ASIDE_THRESH = 20, 0.4, 0.45, 0.3


def render(env, action_fn, steps, seeds, prefix):
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)
    for sd in seeds:
        s = reset(jax.random.PRNGKey(sd))
        states = [s.pipeline_state]
        for t in range(steps):
            s = step(s, action_fn(s.obs, t))
            states.append(s.pipeline_state)
        out = f"{prefix}_seed{sd}.html"
        with open(out, "w") as f:
            f.write(html.render(env.sys, states))
        print(f"saved {out}  final cube_x={float(s.obs[10]):+.2f} "
              f"base_xy=({float(s.obs[0]):+.2f},{float(s.obs[1]):+.2f})", flush=True)


def main():
    # 1) Hallway TASK: scripted navigate -> push -> navigate scaffold.
    he = make_env("tidybot_hallway")

    def scaffold(obs, t):
        base_xy, cube_xy, cube_x, far = obs[0:2], obs[10:12], obs[10], obs[20:22]
        standoff = cube_xy - jnp.array([0.0, STANDOFF])
        if bool(jnp.abs(cube_x) > ASIDE_THRESH):                # lane cleared -> far end
            return base_control(obs[:SD], far)
        if bool(jnp.linalg.norm(base_xy - standoff) < 0.3):     # at standoff -> push
            return push_control(obs[:SD], jnp.array([PUSH_TARGET_X, cube_xy[1]]))
        return base_control(obs[:SD], standoff)                 # else approach standoff
    render(he, scaffold, 400, [0, 1], "viz_hallway_task")

    # 2) BASE skill gathering rollout (navigate env, drive to random targets).
    ne = make_env("tidybot_navigate")
    render(ne, lambda obs, t: base_control(obs[:SD], obs[20:22]), 200, [0, 1],
           "viz_base_skill")

    # 3) PUSH skill gathering rollout (push_aside env, arm-start noise as in collection).
    pe = make_env("tidybot_push_aside")
    pe.arm_noise_scale = 0.8
    render(pe, lambda obs, t: push_control(obs[:SD], obs[20:22]), 200, [0, 1],
           "viz_push_skill")


if __name__ == "__main__":
    main()
