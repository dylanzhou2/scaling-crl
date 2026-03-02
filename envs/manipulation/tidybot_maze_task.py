from envs.manipulation.tidybot_env import TidyBotEnv
import jax
from jax import numpy as jnp

class TidyBotMazeTask(TidyBotEnv):
    """
    Task: Navigate a cluttered maze to a target position while pushing obstacles.
    - Observation: Tidybot state + Goal (3-dim Target Location)
    - Action: 11-dim Whole Body Control
    """

    def _get_xml_path(self):
        # Uses the updated XML including the maze and clutter
        return "envs/assets/stanford_tidybot2/tidybot_cluttered_maze.xml"

    def _set_environment_attributes(self):
        self.env_name = "tidybot_maze_clutter"
        self.episode_length = 500  # Longer for navigation
        self.state_dim = 20        # Core robot observation size
        
        self.base_noise = 0.05
        self.obj_noise = 0.2
        self.goal_dist_threshold = 0.5

    def _get_initial_state(self, rng):
        rng, b_rng, o_rng = jax.random.split(rng, 3)
        
        # Randomize starting base position (x, y, th)
        base_q = jnp.zeros(3) + self.base_noise * jax.random.uniform(b_rng, (3,), minval=-1)
        
        # Default Arm Pose (Folded/Home)
        arm_q = jnp.array([0, 0.26, 3.14, -2.26, 0, 0.96, 1.57])
        
        # Object Spawning (Assuming 'cube' bodies in XML starting at index 11 in qpos)
        # This allows objects to be in different spots every episode
        num_objs = (self.sys.q_size() - 10) // 7
        obj_qs = []
        for _ in range(num_objs):
            rng, sub = jax.random.split(rng)
            obj_qs.append(jax.random.uniform(sub, (7,), minval=-1.0, maxval=1.0))
            
        q = jnp.concatenate([base_q, arm_q] + obj_qs)
        qd = jnp.zeros(self.sys.qd_size())
        return q, qd

    def _get_initial_goal(self, pipeline_state, rng):
        # Target position at the end of the maze
        return jnp.array([4.0, 4.0, 0.0]) 

    def _compute_goal_completion(self, obs, goal):
        # 1. Navigation Reward: Base proximity to maze goal
        base_pos = obs[:2]
        nav_dist = jnp.linalg.norm(base_pos - goal[:2])
        nav_success = nav_dist < self.goal_dist_threshold
        
        # 2. Manipulation Metric: Is the robot near a target object (optional sub-task)
        # (Assuming object positions are part of the goal or task logic)
        
        success = jnp.array(nav_success, dtype=float)
        return success, success, success # Simplified for basic navigation

    def _update_goal_visualization(self, pipeline_state, goal):
        # Move the 'target' ghost object in the XML to the goal position
        updated_q = pipeline_state.q.at[-7:-4].set(goal[:3])
        return pipeline_state.replace(qpos=updated_q)
