"""GPU vs CPU base-motion sanity for tidybot_hallway.

The hallway scaffold froze on the GPU (base_y stayed 0.00, no reset noise, all seeds
identical) while moving fine on CPU. This isolates the cause: does the hallway env
(a) randomize on reset and (b) actually move the base under a constant drive action,
on whatever backend you run it on? Compares against tidybot_push_aside (known to train
on GPU) so a hallway-specific bug stands out.

  uv run diagnose_hallway_motion.py            # both envs, 3 seeds, constant +y drive
"""
import jax
import jax.numpy as jnp
import numpy as np

from train_residual_mab import make_env


def probe(env_id, seeds=(0, 1, 2), steps=100):
    env = make_env(env_id)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    nu = env.action_size
    print(f"\n=== {env_id}  (action_size={nu}) ===", flush=True)
    for sd in seeds:
        s = reset(jax.random.PRNGKey(sd))
        base0 = np.asarray(s.obs[0:3])
        cube0 = np.asarray(s.obs[10:13])
        # Constant drive: +y base, zero arm/gripper. (dim 1 = base_y.)
        a = jnp.zeros(nu).at[1].set(1.0)
        ys, nan = [], False
        for t in range(steps):
            s = step(s, a)
            if t % 20 == 19:
                ys.append(round(float(s.obs[1]), 3))
            nan = nan or bool(jnp.any(jnp.isnan(s.obs)))
        print(f" seed{sd}: init base=({base0[0]:+.3f},{base0[1]:+.3f}) "
              f"cube=({cube0[0]:+.3f},{cube0[1]:+.3f}) | base_y@[20,40,60,80,100]={ys} "
              f"| nan={nan}", flush=True)
    print(" -> base should CLIMB toward the +y goal; init should VARY across seeds "
          "(reset noise). Frozen base_y or identical seeds = the bug.", flush=True)


def main():
    print("JAX devices:", jax.devices(), flush=True)
    probe("tidybot_hallway")
    probe("tidybot_push_aside")  # control: known-good on GPU


if __name__ == "__main__":
    main()
