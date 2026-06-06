from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy
import jax
from jax import numpy as jnp


class TidyBotPushAside(TidyBotPushEasy):
    """Frozen-base non-prehensile push primitive, eased for learnability.

    This subclasses TidyBotPushEasy (the *control* env, left untouched) and applies
    four task-design changes. The reward stays sparse binary throughout — none of
    these is reward shaping; they remove gratuitous exploration difficulty so the
    sparse signal is reachable:

      1. Arm starts at the cube's contact height, just behind it on the push axis,
         so random exploration produces cube contact from t=0. The control env
         starts the EEF ~0.30 m above a floor cube (the pose was copied from a Panda
         env with a 0.37 m table), so the cube never moves under exploration and
         hindsight relabeling sees only "cube at its start" — no signal about pushing.
      2. The goal is a short in-workspace lateral nudge (aside_offset along
         aside_axis from the *actual* cube position), not the control env's ~0.9 m
         cross-workspace push that's infeasible for a frozen base.
      3. Success is xy-only (the control env's goal_z=0.03 vs cube rest z=0.07 eats
         ~40% of the 0.1 m linear success margin).
      4. The gripper is pinned closed (a rigid pusher; one fewer exploration DOF).

    obs = [base(3), arm(7), cube(3), eef(3), eef_vel(3), finger(1)] + cube-xy goal(2)
        = 22.  goal / completion indices = [10, 11] (cube xy).
    """

    # Arm pose that places the fingertips just behind the cube at contact height,
    # found by a kinematic search (fingertip ~ (-0.157, 0.585, 0.095) vs cube at
    # (-0.1, 0.6, 0.07)). Replaces the inherited high pose.
    ARM_Q_CONTACT = jnp.array([-1.71, 1.405, 2.98, -1.237, 0.198, -0.393, 1.351])

    def __init__(self, aside_offset=0.2, zone_radius=0.1, aside_axis=0,
                 gripper_lock=255.0, **kwargs):
        # Stored before super().__init__ so they're available when reset() runs.
        self.aside_offset = aside_offset
        self.zone_radius = zone_radius
        self.aside_axis = aside_axis  # 0 = x (lateral), 1 = y (depth)
        # base_scale defaults to 0.0 (frozen base); gripper pinned closed by default.
        super().__init__(gripper_lock=gripper_lock, **kwargs)

    def _set_environment_attributes(self):
        super()._set_environment_attributes()
        self.env_name = "tidybot_push_aside"
        # xy-only goal: the appended goal is the cube's target xy, and the achieved
        # goal (for HER relabeling, wired in train.py) is obs[10:12] = cube xy.
        self.goal_indices = jnp.array([10, 11])
        self.completion_goal_indices = self.goal_indices

    def _get_initial_state(self, rng):
        # Mirrors the parent but starts the arm at ARM_Q_CONTACT instead of the
        # inherited high pose.
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        q = self.sys.init_q

        cube_q_xy = q[self.cube_q_idx:self.cube_q_idx + 2] + \
            self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        q = q.at[self.cube_q_idx:self.cube_q_idx + 2].set(cube_q_xy)

        arm_q = self.ARM_Q_CONTACT + \
            self.arm_noise_scale * jax.random.uniform(subkey2, [7], minval=-1)
        q = q.at[3:10].set(arm_q)

        return q, jnp.zeros([self.sys.qd_size()])

    def _get_initial_goal(self, pipeline_state, rng):
        # Goal is a fixed lateral nudge from the *actual* (noised) cube position, so
        # the push distance is always exactly aside_offset regardless of cube noise.
        cube_xy = pipeline_state.q[self.cube_q_idx:self.cube_q_idx + 2]
        offset = jnp.zeros(2).at[self.aside_axis].set(self.aside_offset)
        return cube_xy + offset

    def _update_goal_visualization(self, pipeline_state, goal):
        # Place the goal marker at the target xy, at the cube's rest height.
        marker = jnp.array([goal[0], goal[1], 0.07])
        updated_q = pipeline_state.q.at[self.goal_q_idx:self.goal_q_idx + 3].set(marker)
        return pipeline_state.replace(q=updated_q)

    def _compute_goal_completion(self, obs, goal):
        cube_xy = obs[self.completion_goal_indices]
        dist = jnp.linalg.norm(cube_xy - goal[:2])
        success = jnp.array(dist < self.zone_radius, dtype=float)
        success_easy = jnp.array(dist < 3.0 * self.zone_radius, dtype=float)
        success_hard = jnp.array(dist < 0.3 * self.zone_radius, dtype=float)
        return success, success_easy, success_hard
