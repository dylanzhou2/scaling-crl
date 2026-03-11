from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco


class ArmEnvs(PipelineEnv):

  def __init__(self, backend="mjx", **kwargs):
    # Configure environment information (e.g. env name, noise scale, observation dimension, goal indices) and load XML
    self._set_environment_attributes()
    xml_path = self._get_xml_path()
    sys = mjcf.load(xml_path)

    # Configure backend
    sys = sys.tree_replace({
        "opt.timestep": 0.002,
        "opt.iterations": 6,
        "opt.ls_iterations": 12,
    })
    self.n_frames = 25
    kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)

    # Initialize brax PipelineEnv
    if backend != "mjx":
      raise Exception("Use the mjx backend for stability/reasonable speed.")
    super().__init__(sys=sys, backend=backend, **kwargs)
    print("Joint names:", self.sys.link_names)
    print("Position vector size (q):", self.sys.q_size())
    print("Velocity vector size (qd):", self.sys.qd_size())

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""

    rng, subkey = jax.random.split(rng)
    q, qd = self._get_initial_state(subkey)
    pipeline_state = self.pipeline_init(q, qd)
    timestep = 0.0

    rng, subkey1, subkey2 = jax.random.split(rng, 3)
    goal = self._get_initial_goal(pipeline_state, subkey1)
    pipeline_state = self._update_goal_visualization(pipeline_state, goal)
    info = {
        "seed": 0,
        "goal": goal,
        "timestep": 0.0,
        "postexplore_timestep": jax.random.uniform(subkey2),
    }

    obs = self._get_obs(pipeline_state, goal, timestep)
    reward, done, zero = jnp.zeros(3)
    metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
    return State(pipeline_state, obs, reward, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""

    pipeline_state0 = state.pipeline_state
    if "EEF" in self.env_name:
      action = self._convert_action_to_actuator_input_EEF(
          pipeline_state0, action
      )
    else:
      arm_angles = self._get_arm_angles(pipeline_state0)
      action = self._convert_action_to_actuator_input_joint_angle(
          action, arm_angles, delta_control=False
      )

    pipeline_state = self.pipeline_step(pipeline_state0, action)

    if "steps" in state.info.keys():
      seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
    else:
      seed = state.info["seed"]

    timestep = state.info["timestep"] + 1 / self.episode_length
    obs = self._get_obs(pipeline_state, state.info["goal"], timestep)

    success, success_easy, success_hard = self._compute_goal_completion(
        obs, state.info["goal"]
    )
    state.metrics.update(
        success=success, success_easy=success_easy, success_hard=success_hard
    )

    reward = success
    done = 0.0
    info = {**state.info, "timestep": timestep, "seed": seed}

    new_state = state.replace(
        pipeline_state=pipeline_state,
        obs=obs,
        reward=reward,
        done=done,
        info=info,
    )

    if self.env_name == "arm_grasp":
      cube_pos = obs[:3]
      left_finger_goal_pos = cube_pos + jnp.array([0.0375, 0, 0])
      right_finger_goal_pos = cube_pos + jnp.array([-0.0375, 0, 0])
      adjusted_goal = (
          state.info["goal"]
          .at[:6]
          .set(jnp.concatenate([left_finger_goal_pos] + [right_finger_goal_pos]))
      )
      new_state = self.update_goal(new_state, adjusted_goal)

    return new_state

  def update_goal(self, state: State, goal: jax.Array) -> State:
    info = {**state.info, "goal": goal}
    pipeline_state = self._update_goal_visualization(state.pipeline_state, goal)
    return state.replace(pipeline_state=pipeline_state, info=info)

  def _convert_action_to_actuator_input_joint_angle(
      self, action: jax.Array, arm_angles: jax.Array, delta_control=False
  ) -> jax.Array:
    """Converts actions to actuator controls.

    Without mobile base:
      action = [a0, a1, a2, a3, grip]   (5-dim)

    With mobile base:
      action = [base_x, base_y, a0, a1, a2, a3, grip]   (7-dim)

    The trained policy still outputs the old 5-dim arm action; the rollout code
    prepends the 2 manual base controls.
    """

    # Detect whether this env has a mobile base by action length.
    has_mobile_base = action.shape[0] == 7

    if has_mobile_base:
      base_action = action[:2]
      arm_policy_action = action[2:]
    else:
      base_action = None
      arm_policy_action = action

    # Expand 4 arm controls to 7 actual panda joints; joints 3,5,7 fixed.
    arm_action = jnp.array(
        [arm_policy_action[0], arm_policy_action[1], 0,
         arm_policy_action[2], 0, arm_policy_action[3], 0]
    )
    min_value = jnp.array([0.3491, 0, 0, -3.0718, 0, 2.3562, 1.4487])
    max_value = jnp.array([2.7925, 1.48353, 0, -0.0698, 0, 3.7525, 1.4487])

    offset = (min_value + max_value) / 2
    multiplier = (max_value - min_value) / 2

    if delta_control:
      normalized_arm_angles = jnp.where(
          multiplier > 0, (arm_angles - offset) / multiplier, 0
      )
      delta_range = 0.5
      arm_action = normalized_arm_angles + arm_action * delta_range
      arm_action = jnp.clip(arm_action, -1, 1)

    arm_action = offset + arm_action * multiplier

    if self.env_name not in ("arm_reach",):
      gripper_action = jnp.where(
          arm_policy_action[-1] > 0,
          jnp.array([0, 0], dtype=float),
          jnp.array([255, 255], dtype=float),
      )
      converted_action = jnp.concatenate([arm_action, gripper_action])
    else:
      converted_action = arm_action

    # Prepend base actuator controls if present.
    if has_mobile_base:
      converted_action = jnp.concatenate([base_action, converted_action])

    return converted_action

  def _convert_action_to_actuator_input_EEF(
      self, pipeline_state: base.State, action: jax.Array
  ) -> jax.Array:
    eef_index = 2
    current_position = pipeline_state.x.pos[eef_index]
    delta_range = 0.2
    arm_action = current_position + delta_range * jnp.clip(action[:3], -1, 1)

    gripper_action = jnp.where(
        action[-1] > 0,
        jnp.array([0, 0], dtype=float),
        jnp.array([255, 255], dtype=float),
    )

    converted_action = jnp.concatenate([arm_action] + [gripper_action])
    return converted_action

  # Methods to be overridden by specific environments
  def _get_xml_path(self):
    raise NotImplementedError

  def _set_environment_attributes(self):
    raise NotImplementedError

  def _get_initial_state(self, rng):
    raise NotImplementedError

  def _get_initial_goal(self, pipeline_state: base.State, rng):
    raise NotImplementedError

  def _compute_goal_completion(self, obs, goal):
    raise NotImplementedError

  def _update_goal_visualization(
      self, pipeline_state: base.State, goal: jax.Array
  ) -> base.State:
    raise NotImplementedError

  def _get_obs(
      self, pipeline_state: base.State, goal: jax.Array, timestep
  ) -> jax.Array:
    raise NotImplementedError

  def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
    raise NotImplementedError
    