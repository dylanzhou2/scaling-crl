from envs.manipulation.tidybot_envs import TidyBotEnv
import jax
from jax import numpy as jnp


class TidyBotHallway(TidyBotEnv):
    """Composite hallway task: drive the mobile base down a corridor to the far end,
    where a cube blocks the lane. The lane must be cleared (push the cube aside or
    pick it up) for the base to pass.

    This is the env the composition trainer targets (constrained exploration over
    the navigate / push_aside / pick primitives). Reward is sparse: success when the
    base reaches the far-end target. base_scale>0 (mobile); full 11-dim control
    (base + arm + gripper), so the policy can both drive and manipulate.

    obs = [base(3), arm(7), cube(3), eef(3), eef_vel(3), finger(1)] + base-xy goal(2)
        = 22.  goal / completion indices = [0, 1] (base xy, far end). The cube is in
    the state (dims 10:13), so the policy can see the obstacle.

    KNOWN LIMITATION (plan D7): a straight corridor lets an unconstrained policy
    bulldoze the cube forward to the goal rather than clearing it aside. Anti-bulldoze
    hardening (cube friction/mass, a lateral alcove, or success conditioned on the
    cube being out of lane) is a follow-up; for now the corridor + blocking cube set
    up the navigate->clear->navigate task for the composition experiment.
    """

    ARM_Q_FOLD = jnp.array([0.0, -1.0, 0.0, -2.0, 0.0, 1.0, 0.0])

    def __init__(self, base_scale=0.2, success_radius=0.4, goal_xy=(0.0, 2.5),
                 aside_thresh=0.3, lift_thresh=0.15,
                 cube_noise_scale=0.25, base_noise_scale=0.05, **kwargs):
        self.success_radius = success_radius
        self.goal_xy = jnp.array(goal_xy)
        # Anti-bulldoze: success requires the cube CLEARED from the lane, not just
        # the base at the goal. A small free cube can't be force/mass-tuned to block
        # the base, so we reject the bulldoze solution at the reward level instead:
        # the cube must be pushed aside (|x| > aside_thresh) or lifted (z > lift_thresh).
        # Clean here because the hallway is a single task with constrained exploration
        # over primitives, NOT CRL/HER goal-reaching.
        self.aside_thresh = aside_thresh
        self.lift_thresh = lift_thresh
        self.cube_noise_scale = cube_noise_scale
        self.base_noise_scale = base_noise_scale
        super().__init__(base_scale=base_scale, **kwargs)

    def _get_xml_path(self):
        return "envs/assets/tidybot_hallway.xml"

    def _set_environment_attributes(self):
        self.env_name = "tidybot_hallway"
        self.episode_length = 500
        self.cube_q_idx = 18
        self.goal_q_idx = 25
        self.goal_indices = jnp.array([0, 1])  # base xy
        self.completion_goal_indices = self.goal_indices
        self.state_dim = 20
        self.arm_noise_scale = 0.0

    def _get_initial_state(self, rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        q = self.sys.init_q
        # Base at the near end (origin) with small noise.
        base_xy = q[:2] + self.base_noise_scale * jax.random.uniform(k1, [2], minval=-1)
        q = q.at[:2].set(base_xy)
        # Arm tucked at start; the policy/composition lowers it to push.
        q = q.at[3:10].set(self.ARM_Q_FOLD)
        # Cube mid-corridor (XML default 0, 1.2) with optional noise.
        cube_xy = q[self.cube_q_idx:self.cube_q_idx + 2] + \
            self.cube_noise_scale * jax.random.uniform(k2, [2], minval=-1)
        q = q.at[self.cube_q_idx:self.cube_q_idx + 2].set(cube_xy)
        return q, jnp.zeros([self.sys.qd_size()])

    def _get_initial_goal(self, pipeline_state, rng):
        return self.goal_xy

    def _update_goal_visualization(self, pipeline_state, goal):
        marker = jnp.array([goal[0], goal[1], 0.05])
        updated_q = pipeline_state.q.at[self.goal_q_idx:self.goal_q_idx + 3].set(marker)
        return pipeline_state.replace(q=updated_q)

    def _compute_goal_completion(self, obs, goal):
        base_xy = obs[self.completion_goal_indices]
        cube_x = obs[10]
        cube_z = obs[12]
        base_dist = jnp.linalg.norm(base_xy - goal[:2])
        base_at_goal = base_dist < self.success_radius
        # Lane is cleared iff the cube was pushed aside or lifted out.
        lane_clear = jnp.logical_or(
            jnp.abs(cube_x) > self.aside_thresh, cube_z > self.lift_thresh
        )
        # Reward: base at goal AND lane cleared (rejects bulldoze).
        success = jnp.array(jnp.logical_and(base_at_goal, lane_clear), dtype=float)
        # Diagnostics: success_easy ignores the cube (reveals bulldozing if it
        # exceeds success); success_hard tightens the base tolerance.
        success_easy = jnp.array(base_at_goal, dtype=float)
        success_hard = jnp.array(
            jnp.logical_and(base_dist < 0.5 * self.success_radius, lane_clear),
            dtype=float,
        )
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
