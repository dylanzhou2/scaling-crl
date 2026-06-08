"""Render an end-to-end hallway scaffold: scripted navigate -> CRL push -> navigate.

Loads a trained tidybot_push_aside CRL actor and deploys it in the hallway. The
push policy was trained base-at-origin / base-frozen / arm-at-contact, so to use it
in the hallway we:
  * feed it a BASE-RELATIVE obs (subtract the base position from cube/eef so the cube
    appears where the policy saw it in push_aside),
  * HOLD the base during the push (zero base action dims), and
  * LOWER the arm toward the contact pose while approaching the standoff.

Phases: approach standoff (+lower arm) -> CRL push until the lane clears -> drive to
the far end.

  uv run render_hallway_scaffold.py --checkpoint PATH/..._push_aside_..._final.pkl
"""
import argparse

from brax.io import html
from flax import linen as nn
import jax
import jax.numpy as jnp

from train import Actor, load_params
from train_residual_mab import make_env
from envs.scripted_controllers import base_control, ARM_Q_CONTACT

SD, STANDOFF, ASIDE, ASIDE_THRESH = 20, 0.6, 0.2, 0.3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = p.parse_args()

    sv = load_params(args.checkpoint.replace("_final.pkl", "_args.pkl"))
    actor = Actor(action_size=11, network_width=sv.actor_network_width,
                  network_depth=sv.actor_depth,
                  skip_connections=sv.actor_skip_connections, use_relu=sv.use_relu)
    _a, push_params, _c = load_params(args.checkpoint)

    @jax.jit
    def push_action(obs):
        s = obs[:SD]
        b = jnp.array([s[0], s[1], 0.0])                    # base xy offset
        push_obs = jnp.concatenate([
            jnp.array([0.0, 0.0, s[2]]),                    # base at origin
            s[3:10],                                        # arm joints
            s[10:13] - b,                                   # cube (base-relative)
            s[13:16] - b,                                   # eef  (base-relative)
            s[16:19], s[19:20],                             # eef_vel, finger
            (s[10:12] - b[:2]) + jnp.array([ASIDE, 0.0]),   # push goal (base-rel)
        ])
        mean, _ = actor.apply(push_params, push_obs[None])
        a = nn.tanh(mean)[0]
        return a.at[0:3].set(0.0)                           # hold base during push

    def nav_lower(obs, goal):
        a = base_control(obs[:SD], goal)
        a = a.at[3:10].set(jnp.clip(ARM_Q_CONTACT - obs[3:10], -1.0, 1.0))  # lower arm
        return a.at[10].set(-1.0)

    env = make_env("tidybot_hallway")
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)

    for sd in args.seeds:
        s = reset(jax.random.PRNGKey(sd))
        states = [s.pipeline_state]
        ph = {"approach": 0, "push": 0, "far": 0}
        for t in range(args.steps):
            obs = s.obs
            base_xy, cube_xy, cube_x, far = obs[0:2], obs[10:12], obs[10], obs[20:22]
            standoff = cube_xy - jnp.array([0.0, STANDOFF])
            if bool(jnp.abs(cube_x) > ASIDE_THRESH):
                a = base_control(obs[:SD], far); ph["far"] += 1
            elif bool(jnp.linalg.norm(base_xy - standoff) < 0.25):
                a = push_action(obs); ph["push"] += 1
            else:
                a = nav_lower(obs, standoff); ph["approach"] += 1
            s = step(s, a)
            states.append(s.pipeline_state)
        out = f"viz_scaffold_seed{sd}.html"
        with open(out, "w") as f:
            f.write(html.render(env.sys, states))
        print(f"seed{sd}: phases={ph}  final cube_x={float(s.obs[10]):+.2f} "
              f"base_y={float(s.obs[1]):+.2f} (cleared if |cube_x|>0.3, far end ~2.5) "
              f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
