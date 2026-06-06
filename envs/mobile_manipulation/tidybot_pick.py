from envs.manipulation.tidybot_envs import TidyBotEnv
import jax
from jax import numpy as jnp


class TidyBotPick(TidyBotEnv):
    """Scoped top-down pick primitive: grasp a fixed cube and lift it.

    Frozen base (base_scale=0); the gripper is policy-controlled (NOT locked) so the
    policy must close it to grasp. Scoped per the plan: fixed cube, top-down
    parallel-jaw. Success = the cube reaches the lifted goal (cube + lift_height),
    which is only achievable by actually grasping and lifting it.

    obs = [base(3), arm(7), cube(3), eef(3), eef_vel(3), finger(1)] + cube-xyz goal(3)
        = 23.  goal / completion indices = [10, 11, 12] (cube xyz).
    """

    # Arm pose placing the open gripper around the SETTLED cube, pointing down
    # (kinematic search vs the cube's rest height 0.03: fingertips ~(-0.12, 0.60,
    # 0.02) straddling the cube, bracelet ~0.19 m above them).
    ARM_Q_GRASP = jnp.array([-1.882, 1.645, 3.582, -1.41, 0.875, 0.763, -0.245])

    def __init__(self, base_scale=0.0, lift_height=0.15, lift_thresh=0.08,
                 cube_rest_z=0.03, cube_noise_scale=0.0, **kwargs):
        self.lift_height = lift_height
        self.lift_thresh = lift_thresh
        self.cube_rest_z = cube_rest_z
        self.cube_noise_scale = cube_noise_scale
        # gripper NOT locked: the policy controls action[10] to grasp.
        super().__init__(base_scale=base_scale, **kwargs)

    def _get_xml_path(self):
        # Pick-specific scene: higher pad friction + higher gripper closure force
        # for a firm grasp (the shared push scene caps gripper force at 5 N).
        return "envs/assets/tidybot_pick.xml"

    def _set_environment_attributes(self):
        self.env_name = "tidybot_pick"
        self.episode_length = 200
        self.cube_q_idx = 18
        self.goal_q_idx = 25
        self.goal_indices = jnp.array([10, 11, 12])  # cube xyz
        self.completion_goal_indices = self.goal_indices
        self.state_dim = 20
        self.arm_noise_scale = 0.0

    def _get_initial_state(self, rng):
        rng, k = jax.random.split(rng)
        q = self.sys.init_q
        arm_q = self.ARM_Q_GRASP + self.arm_noise_scale * jax.random.uniform(k, [7], minval=-1)
        q = q.at[3:10].set(arm_q)
        # Start the cube at its settled rest height (XML inits it 0.04 m high).
        q = q.at[self.cube_q_idx + 2].set(self.cube_rest_z)
        if self.cube_noise_scale > 0:
            rng, k2 = jax.random.split(rng)
            cube_xy = q[self.cube_q_idx:self.cube_q_idx + 2] + \
                self.cube_noise_scale * jax.random.uniform(k2, [2], minval=-1)
            q = q.at[self.cube_q_idx:self.cube_q_idx + 2].set(cube_xy)
        return q, jnp.zeros([self.sys.qd_size()])

    def _get_initial_goal(self, pipeline_state, rng):
        cube = pipeline_state.q[self.cube_q_idx:self.cube_q_idx + 3]
        return cube + jnp.array([0.0, 0.0, self.lift_height])

    def _update_goal_visualization(self, pipeline_state, goal):
        updated_q = pipeline_state.q.at[self.goal_q_idx:self.goal_q_idx + 3].set(goal[:3])
        return pipeline_state.replace(q=updated_q)

    def _compute_goal_completion(self, obs, goal):
        cube_pos = obs[10:13]
        dist = jnp.linalg.norm(cube_pos - goal[:3])
        lifted = cube_pos[2] - self.cube_rest_z
        # Success = cube reached the lifted goal (only possible by grasping + lifting).
        success = jnp.array(dist < 0.08, dtype=float)
        success_easy = jnp.array(lifted > 0.3 * self.lift_thresh, dtype=float)  # any lift
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard

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
