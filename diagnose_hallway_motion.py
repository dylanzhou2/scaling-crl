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


def probe(env_id, seeds=(0, 1, 2), steps=100, drive=1.0):
    env = make_env(env_id)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    nu = env.action_size
    print(f"\n=== {env_id}  (action_size={nu}, drive={drive}) ===", flush=True)
    for sd in seeds:
        s = reset(jax.random.PRNGKey(sd))
        base0 = np.asarray(s.obs[0:3])
        cube0 = np.asarray(s.obs[10:13])
        # Constant drive: +y base, zero arm/gripper. (dims 0:3 = base x, y, theta.)
        a = jnp.zeros(nu).at[1].set(drive)
        ys, ths, blowup = [], [], -1
        for t in range(steps):
            s = step(s, a)
            by, bth = float(s.obs[1]), float(s.obs[2])
            finite = bool(jnp.all(jnp.isfinite(s.obs)))
            if t < 30 and t % 5 == 4:
                ys.append(round(by, 2)); ths.append(round(bth, 2))
            # First step where the state diverges (non-finite or |yaw| runs away).
            if blowup < 0 and (not finite or abs(bth) > 1.0 or abs(by) > 50.0):
                blowup = t
        print(f" seed{sd}: init base=({base0[0]:+.3f},{base0[1]:+.3f}) "
              f"cube=({cube0[0]:+.3f},{cube0[1]:+.3f}) | base_y@[5,10,15,20,25,30]={ys} "
              f"| base_th@same={ths} | first_blowup_step={blowup}", flush=True)
    print(" -> base_y should CLIMB smoothly and base_th should stay ~0. A non-zero yaw "
          "that runs away (then base_y -> overflow) is the inertia bug; "
          "first_blowup_step=-1 means stable.", flush=True)


def main():
    print("JAX devices:", jax.devices(), flush=True)
    probe("tidybot_hallway")
    probe("tidybot_push_aside")  # control: known-good on GPU


if __name__ == "__main__":
    main()
