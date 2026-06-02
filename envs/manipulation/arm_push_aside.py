from brax import base
from envs.manipulation.arm_push_hard import ArmPushHard
import jax
from jax import numpy as jnp

"""
Push-Aside: an *adjacent task* to arm_push_hard, used to probe whether the frozen
arm_push_hard CRL policy can be reused AS IS (+ a Mahalanobis Action Barrier
residual) when the goal is pushed *further aside* than it was ever trained.

Same scene / robot / obs / action / dynamics as ArmPushHard (we subclass it). The
only changes:

  * The goal is placed deterministically at a controllable rightward offset along
    the lateral (x) axis: goal_x = 0.25 + aside_offset. The frozen policy was
    trained on a strictly rightward push (cube starts on the blue bin at x=-0.25,
    goal lives in the red bin x in [0, 0.5]), so:
        - aside_offset in [-0.25, +0.25]  ->  goal_x in [0, 0.5]  (in-distribution)
        - aside_offset  >  +0.25          ->  goal_x  >  0.5      (out-of-distribution)
    `aside_offset` is therefore the experiment's swept independent variable.

  * Success is a looser "in the zone" criterion (dist < zone_radius) instead of
    the coordinate-tight 0.1 m used by arm_push_hard, since the task is "land the
    cube in a region", not "hit an exact point". This makes the env reward equal
    region success, which is what train_residual_mab.py's eval measures.

Observation/action spaces are identical to ArmPushHard (17-dim obs + 3-dim goal,
5-dim action), so the frozen arm_push_hard checkpoint loads and runs unchanged.
"""


class ArmPushAside(ArmPushHard):

  def __init__(
      self,
      backend="mjx",
      aside_offset=0.15,
      zone_radius=0.12,
      aside_axis=0,
      **kwargs,
  ):
    # Set before super().__init__, which immediately calls
    # _set_environment_attributes() and (on reset) _get_initial_goal().
    self.aside_offset = aside_offset
    self.zone_radius = zone_radius
    self.aside_axis = aside_axis  # 0 = x (lateral / left-right), 1 = y (depth)
    super().__init__(backend=backend, **kwargs)

  def _set_environment_attributes(self):
    self.env_name = "arm_push_aside"  # must NOT contain "EEF" -> joint-angle control
    self.episode_length = 100  # a touch longer than push_hard (80) for further travel

    self.goal_indices = jnp.array([0, 1, 2])  # Cube position
    self.completion_goal_indices = jnp.array([0, 1, 2])  # Identical
    self.state_dim = 17

    self.arm_noise_scale = 0
    self.cube_noise_scale = 0.05  # small: keep the cube start ~centered/consistent
    self.goal_noise_scale = 0.1   # small y-jitter of the goal within the zone

  def _get_initial_goal(self, pipeline_state: base.State, rng):
    rng, subkey = jax.random.split(rng)
    # Trained red-bin center is (0.25, 0.65, 0.03); shift rightward by aside_offset.
    base_center = jnp.array([0.25, 0.65, 0.03])
    base_center = base_center.at[self.aside_axis].add(self.aside_offset)
    # Jitter only in depth (y) so each offset is a crisp lateral-distance condition.
    jitter = jnp.array([0.0, self.goal_noise_scale, 0.0]) * jax.random.uniform(
        subkey, [3], minval=-1
    )
    return base_center + jitter

  def _compute_goal_completion(self, obs, goal):
    current_cube_pos = obs[self.completion_goal_indices]
    goal_pos = goal[:3]
    dist = jnp.linalg.norm(current_cube_pos - goal_pos)

    # Looser, region-style success. Reward = success (set in ArmEnvs.step).
    success = jnp.array(dist < self.zone_radius, dtype=float)
    success_easy = jnp.array(dist < 2.0 * self.zone_radius, dtype=float)
    success_hard = jnp.array(dist < 0.1, dtype=float)  # the push_hard coordinate-tight bar
    return success, success_easy, success_hard
