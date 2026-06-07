from envs.manipulation.tidybot_envs import TidyBotEnv
import jax
from jax import numpy as jnp


class TidyBotNavigate(TidyBotEnv):
    """Base-only navigation CRL primitive: drive the holonomic base to a target xy.

    A demo-free, sparse-reward CRL navigation primitive (NOT the maze's dense
    -dist / early-done formulation, which breaks fixed-length HER relabeling). The
    arm is pinned to a tucked fold pose and the base is mobile (base_scale>0), so
    only the 3 base action dims matter — but the action vector stays 11-dim and the
    state stays the shared 20-dim TidyBot layout, so this primitive is
    obs/action-compatible with push_aside/pick for composition.

    obs = [base(3), arm(7), cube(3), eef(3), eef_vel(3), finger(1)] + base-xy goal(2)
        = 22.  goal / completion indices = [0, 1] (base xy). The cube is parked far
        away so it never interferes with navigation.
    """

    # Tucked arm pose held throughout (compact, won't catch on obstacles).
    ARM_Q_FOLD = jnp.array([0.0, -1.0, 0.0, -2.0, 0.0, 1.0, 0.0])
    CUBE_PARK = jnp.array([5.0, 5.0])  # park the cube out of the way

    def __init__(self, base_scale=0.2, success_radius=0.5,
                 target_center=(0.0, 1.0), target_scale=1.5, base_noise_scale=0.1,
                 **kwargs):
        self.success_radius = success_radius
        self.target_center = jnp.array(target_center)
        self.target_scale = target_scale
        self.base_noise_scale = base_noise_scale
        super().__init__(base_scale=base_scale, **kwargs)

    def _get_xml_path(self):
        # Reuse the push scene; we ignore/park the cube. (A dedicated flat scene is
        # a possible cleanup, but this avoids a new XML.)
        return "envs/assets/tidybot_push_easy.xml"

    def _set_environment_attributes(self):
        self.env_name = "tidybot_navigate"
        self.episode_length = 300
        self.cube_q_idx = 18
        self.goal_q_idx = 25
        self.goal_indices = jnp.array([0, 1])  # base xy
        self.completion_goal_indices = self.goal_indices
        self.state_dim = 20
        self.arm_noise_scale = 0.0

    def _get_initial_state(self, rng):
        rng, subkey = jax.random.split(rng)
        q = self.sys.init_q
        # Base near origin with small noise.
        base_xy = q[:2] + self.base_noise_scale * jax.random.uniform(subkey, [2], minval=-1)
        q = q.at[:2].set(base_xy)
        # Arm tucked; cube parked far away.
        q = q.at[3:10].set(self.ARM_Q_FOLD)
        q = q.at[self.cube_q_idx:self.cube_q_idx + 2].set(self.CUBE_PARK)
        return q, jnp.zeros([self.sys.qd_size()])

    def _get_initial_goal(self, pipeline_state, rng):
        rng, subkey = jax.random.split(rng)
        return self.target_center + self.target_scale * jax.random.uniform(
            subkey, [2], minval=-1
        )

    def _update_goal_visualization(self, pipeline_state, goal):
        marker = jnp.array([goal[0], goal[1], 0.05])
        updated_q = pipeline_state.q.at[self.goal_q_idx:self.goal_q_idx + 3].set(marker)
        return pipeline_state.replace(q=updated_q)

    def _compute_goal_completion(self, obs, goal):
        base_xy = obs[self.completion_goal_indices]
        dist = jnp.linalg.norm(base_xy - goal[:2])
        success = jnp.array(dist < self.success_radius, dtype=float)
        success_easy = jnp.array(dist < 2.0 * self.success_radius, dtype=float)
        success_hard = jnp.array(dist < 0.4 * self.success_radius, dtype=float)
        return success, success_easy, success_hard

    def _convert_action_to_actuator_input_joint_angle(
        self, action, arm_angles, delta_control: bool = True
    ) -> jax.Array:
        # Base-only control: base from action[:3]*base_scale, arm held at the fold
        # pose (absolute position target), gripper irrelevant (held closed).
        action = jnp.clip(action, -1.0, 1.0)
        base_current = arm_angles[:3]
        # Holonomic base: translate in world x/y to reach any target. Hold heading
        # at 0 (face +y) instead of letting the policy spin joint_th freely (heading
        # isn't in the goal, so an unconstrained base just rotates). A fixed heading
        # also matches push_aside's frozen-base pose for a clean composition handoff.
        base_xy = base_current[:2] + action[:2] * self.base_scale
        base_th = jnp.zeros(1)
        base_action = jnp.concatenate([base_xy, base_th])
        arm_action = self.ARM_Q_FOLD
        gripper_action = jnp.array([255.0])
        return jnp.concatenate([base_action, arm_action, gripper_action])

    def _get_obs(self, pipeline_state, goal, timestep):
        base_q = pipeline_state.q[:3]
        arm_q = pipeline_state.q[3:10]
        cube_pos = pipeline_state.q[self.cube_q_idx:self.cube_q_idx + 3]
        eef_pos = pipeline_state.x.pos[self.eef_index]
        eef_vel = pipeline_state.xd.vel[self.eef_index]
        finger_dist = jnp.linalg.norm(
            pipeline_state.x.pos[11] - pipeline_state.x.pos[15], keepdims=True
        )
        return jnp.concatenate(
            [base_q, arm_q, cube_pos, eef_pos, eef_vel, finger_dist, goal]
        )
