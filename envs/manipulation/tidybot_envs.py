from brax import base
from brax.envs.base import State
from brax.io import mjcf
from envs.manipulation.arm_envs import ArmEnvs
import jax
from jax import numpy as jnp

class TidybotEnv(ArmEnvs):
  """
  Tidybot Whole-Body Environment: Locomotion (3-DOF) + Manipulation (7-DOF).
  - Action space (8-dim): 
      [base_x, base_y, base_theta, arm_q1, arm_q2, arm_q4, arm_q6, gripper]
  - Observation space (20-dim): 
      [base_q(3), arm_q(7), eef_pos(3), eef_vel(3), gripper_dist(1)] + goal(3)
  """

  def _get_xml_path(self):
    return "envs/assets/stanford_tidybot2/tidybot.xml"

  @property
  def action_size(self) -> int:
    return 8  # 3 (base) + 4 (arm joints) + 1 (gripper)

  def _set_environment_attributes(self):
    self.env_name = "tidybot_whole_body"
    self.episode_length = 100
    self.state_dim = 20  # Observations without goal
    
    # Noise scales for randomization
    self.base_noise_scale = 0.1
    self.arm_noise_scale = 0.05
    self.goal_noise_scale = 0.3

  def _get_initial_state(self, rng):
    rng, b_rng, a_rng = jax.random.split(rng, 3)
    
    # Initialize Base (x, y, theta)
    base_q = jnp.zeros(3) + self.base_noise_scale * jax.random.uniform(b_rng, (3,), minval=-1)
    
    # Initialize Arm (Joints 1-7)
    arm_q_default = jnp.array([1.571, 0.742, 0, -1.571, 0, 3.054, 1.449])
    arm_q = arm_q_default + self.arm_noise_scale * jax.random.uniform(a_rng, (7,), minval=-1)
    
    # Gripper initialization (multi-joint 4-bar linkage)
    gripper_q = jnp.zeros(self.sys.q_size() - 10) 
    
    q = jnp.concatenate([base_q, arm_q, gripper_q])
    qd = jnp.zeros(self.sys.qd_size())
    return q, qd

  def _get_initial_goal(self, pipeline_state: base.State, rng):
    # Goal in a box relative to the starting area
    goal = jnp.array([0.5, 0.5, 0.3]) + self.goal_noise_scale * jax.random.uniform(rng, (3,), minval=-1)
    return goal

  def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
    # Base state: joints 0, 1, 2
    base_q = pipeline_state.q[:3]
    
    # Arm state: joints 3 to 10
    arm_q = pipeline_state.q[3:10]
    
    # End-Effector (EEF) from bracelet_link (Body index 7)
    eef_index = 7
    eef_pos = pipeline_state.x.pos[eef_index]
    eef_vel = pipeline_state.xd.vel[eef_index]
    
    # Finger distance calculation
    left_finger_pos = pipeline_state.x.pos[9]
    right_finger_pos = pipeline_state.x.pos[13]
    finger_dist = jnp.linalg.norm(right_finger_pos - left_finger_pos, keepdims=True)

    return jnp.concatenate([base_q, arm_q, eef_pos, eef_vel, finger_dist, goal])

  def step(self, state: State, action: jax.Array) -> State:
    """Integrated locomotion and manipulation step."""
    pipeline_state0 = state.pipeline_state
    
    # 1. Base Locomotion (dims 0, 1, 2)
    # Mapping [-1, 1] to a target movement range (e.g., +/- 0.5m or radians)
    base_target = action[:3] * 0.5 + pipeline_state0.q[:3]
    
    # 2. Arm Manipulation (dims 3, 4, 5, 6)
    arm_angles = self._get_arm_angles(pipeline_state0)
    # Using the existing joint angle conversion
    arm_action_mapped = self._convert_action_to_actuator_input_joint_angle(
        action[3:8], arm_angles, delta_control=False
    )
    
    # 3. Concatenate Base Position Targets + Arm/Gripper Actuator Targets
    # Note: tidybot.xml uses position actuators for the base joints
    full_action = jnp.concatenate([base_target, arm_action_mapped])
    
    # Step the simulation
    pipeline_state = self.pipeline_step(pipeline_state0, full_action)
    
    # Standard State Update
    timestep = state.info["timestep"] + 1 / self.episode_length
    obs = self._get_obs(pipeline_state, state.info["goal"], timestep)
    
    success, success_easy, success_hard = self._compute_goal_completion(obs, state.info["goal"])
    state.metrics.update(success=success, success_easy=success_easy, success_hard=success_hard)
    
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=success, info={**state.info, "timestep": timestep}
    )

  def _compute_goal_completion(self, obs, goal):
    # Success based on EEF distance to goal
    eef_pos = obs[10:13]
    dist = jnp.linalg.norm(eef_pos - goal)
    return (jnp.array(dist < 0.1, dtype=float), 
            jnp.array(dist < 0.3, dtype=float), 
            jnp.array(dist < 0.03, dtype=float))

  def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
    return pipeline_state.q[3:10]

  def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
    # Goal markers are usually at the start of qpos for visualization objects
    return pipeline_state # Placeholder: requires specific goal geom in XML
