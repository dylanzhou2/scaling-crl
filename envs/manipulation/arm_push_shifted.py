import os

import jax
from jax import numpy as jnp

from envs.manipulation.arm_push_hard import ArmPushHard

"""
Push-Shifted: arm_push_hard with a RELOCATABLE goal center, for out-of-distribution
goal experiments (does a goal shift break the vanilla CRL policy while the MAB
residual recovers it?).

Everything is inherited from ArmPushHard (same cube XML, obs/goal layout, action
space) so an arm_push_hard checkpoint applies directly. The only change is the goal
center, read from the env var ARM_PUSH_GOAL_CENTER ("x,y,z"); it defaults to the
original red-region center, so with the var unset this env == arm_push_hard (with
tighter goal noise). Goal noise is reduced from 0.25 -> 0.1 so the shifted target is
sharp and the vanilla-vs-MAB contrast is clean.

The training goal center is [0.25, 0.65, 0.03] and the cube starts near
[-0.25, 0.65], so the trained push is ~0.5 m in +x. Shifting the goal further in
+x (e.g. 0.45-0.65) puts it past the trained goal range.

Default goal here is [0.45, 0.65, 0.03]: a FAR target (0.2 m past the trained
center) that is still inside the arm's workspace (~0.8 m horizontal reach at table
height => usable |x| < ~0.47 at y=0.65). Goal noise is 0 so it is a single fixed
point, cleanest for a visualization rollout. Override with the env var, e.g.
ARM_PUSH_GOAL_CENTER="0.5,0.6,0.03", to move it without editing this file.
"""


def _parse_goal_center(default=(0.45, 0.65, 0.03)):
  raw = os.environ.get("ARM_PUSH_GOAL_CENTER", "").strip()
  if not raw:
    return jnp.array(default)
  vals = [float(x) for x in raw.split(",")]
  assert len(vals) == 3, f"ARM_PUSH_GOAL_CENTER must be 'x,y,z', got {raw!r}"
  return jnp.array(vals)


class ArmPushShifted(ArmPushHard):

  def _set_environment_attributes(self):
    super()._set_environment_attributes()
    self.env_name = "arm_push_shifted"
    self.goal_center = _parse_goal_center()
    self.goal_noise_scale = 0.0  # fixed goal -> single sharp target for visualization

  def _get_initial_goal(self, pipeline_state, rng):
    rng, subkey = jax.random.split(rng)
    return self.goal_center + jnp.array(
        [self.goal_noise_scale, self.goal_noise_scale, 0]
    ) * jax.random.uniform(subkey, [3], minval=-1)
