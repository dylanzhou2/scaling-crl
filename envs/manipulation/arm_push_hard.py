from brax import base
from envs.manipulation.arm_envs import ArmEnvs
import jax
from jax import numpy as jnp

"""
Push-Hard with mobile base.

Observation space: 17-dim obs + 3-dim goal.
Policy action space: 5-dim (same trained arm policy output).
Env action space:    7-dim = [base_x, base_y, arm_policy_5d].

q layout:
  [0]     base_x
  [1]     base_y
  [2:11]  arm joints + fingers (9)
  [11:18] cube freejoint (7)
  [18:25] goal_marker freejoint (7)

link_names:
  ['mobile_base', 'link1', 'link2', 'link3', 'link4',
   'link5', 'link6', 'link7', 'left_finger', 'right_finger',
   'cube', 'goal_marker']
"""


class ArmPushHard(ArmEnvs):

  def _get_xml_path(self):
    return "envs/assets/panda_push_hard.xml"

  @property
  def action_size(self) -> int:
    # 2 base controls + original 5-dim arm policy action
    return 7

  def _set_environment_attributes(self):
    self.env_name = "arm_push_hard"
    self.episode_length = 80

    self.goal_indices = jnp.array([0, 1, 2])  # cube xyz inside obs
    self.completion_goal_indices = jnp.array([0, 1, 2])
    self.state_dim = 17

    self.arm_noise_scale = 0
    self.cube_noise_scale = 0.25
    self.goal_noise_scale = 0.25

  def _get_initial_state(self, rng):
    rng, subkey1, subkey2 = jax.random.split(rng, 3)

    # Since link0 is now mounted at pos="0 0 0.05",
    # base_q = [0, 0] means the robot starts centered on the base.
    base_q = jnp.array([0.0, 0.0])

    # arm + fingers occupy q[2:11]
    arm_q_default = jnp.array(
        [1.571, 0.742, 0, -1.571, 0, 3.054, 1.449, 0.04, 0.04]
    )
    arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(
        subkey2, [9], minval=-1
    )

    # cube freejoint occupies q[11:18]
    cube_q_xy = self.sys.init_q[11:13] + self.cube_noise_scale * jax.random.uniform(
        subkey1, [2], minval=-1
    )
    cube_q_remaining = self.sys.init_q[13:18]

    # goal marker freejoint occupies q[18:25]
    target_q = self.sys.init_q[18:25]

    q = jnp.concatenate([base_q, arm_q, cube_q_xy, cube_q_remaining, target_q])
    qd = jnp.zeros([self.sys.qd_size()])
    return q, qd

  def _get_initial_goal(self, pipeline_state: base.State, rng):
    rng, subkey = jax.random.split(rng)
    cube_goal_pos = jnp.array([0.25, 0.65, 0.03]) + jnp.array(
        [self.goal_noise_scale, self.goal_noise_scale, 0]
    ) * jax.random.uniform(subkey, [3], minval=-1)
    return cube_goal_pos

  def _compute_goal_completion(self, obs, goal):
    current_cube_pos = obs[self.completion_goal_indices]
    goal_pos = goal[:3]
    dist = jnp.linalg.norm(current_cube_pos - goal_pos)

    success = jnp.array(dist < 0.1, dtype=float)
    success_easy = jnp.array(dist < 0.3, dtype=float)
    success_hard = jnp.array(dist < 0.03, dtype=float)
    return success, success_easy, success_hard

  def _update_goal_visualization(
      self, pipeline_state: base.State, goal: jax.Array
  ) -> base.State:
    # goal marker xyz lives at q[18:21]
    updated_q = pipeline_state.q.at[18:21].set(goal[:3])

    # Re-run forward kinematics so rendering matches q
    updated_pipeline_state = self.pipeline_init(updated_q, pipeline_state.qd)
    return updated_pipeline_state

  def _get_obs(
      self, pipeline_state: base.State, goal: jax.Array, timestep
  ) -> jax.Array:
    """Observation space (17-dim)

    - q_subset (10-dim): 3-dim cube position, 7-dim arm joint angles
    - End-effector (6-dim): position and velocity
    - Fingers (1-dim): finger distance
    - Goal (3-dim): cube target xyz
    """

    # cube xyz = q[11:14], arm joints (not fingers) = q[2:9]
    q_indices = jnp.array([11, 12, 13, 2, 3, 4, 5, 6, 7, 8])
    q_subset = pipeline_state.q[q_indices]

    # link indices after adding mobile base:
    # 0 mobile_base, 1 link1, ..., 7 link7, 8 left_finger, 9 right_finger
    eef_index = 7
    eef_x_pos = pipeline_state.x.pos[eef_index]
    eef_xd_vel = pipeline_state.xd.vel[eef_index]

    left_finger_index = 8
    right_finger_index = 9
    left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
    right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
    finger_distance = jnp.linalg.norm(
        right_finger_x_pos - left_finger_x_pos
    )[None]

    return jnp.concatenate(
        [q_subset, eef_x_pos, eef_xd_vel, finger_distance, goal]
    )

  def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
    # 7 arm joints are q[2:9]
    return pipeline_state.q[2:9]
    