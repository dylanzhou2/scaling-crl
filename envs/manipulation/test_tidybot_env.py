import jax
import jax.numpy as jnp
from envs.manipulation.tidybot_envs import TidybotEnv

def test_tidybot_env():
    # 1. Initialize Environment
    env = TidybotEnv(backend="mjx")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    print(f"Observation Shape: {state.obs.shape}")
    assert state.obs.shape[0] == 23, "Obs shape mismatch (20 state + 3 goal)"
    
    # 2. Test Physical Movement
    # Action: Move base forward in X, move arm joints
    action = jnp.array([0.5, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, -1.0])
    
    new_state = jit_step(state, action)
    
    # Verify Base Movement
    base_pos_start = state.obs[:3]
    base_pos_end = new_state.obs[:3]
    print(f"Base Start: {base_pos_start} -> End: {base_pos_end}")
    
    if jnp.allclose(base_pos_start, base_pos_end):
        print("FAILED: No base movement detected.")
    else:
        print("SUCCESS: Base moved correctly.")

    # 3. Test Reward Calculation
    print(f"Initial Reward: {state.reward}, Step Reward: {new_state.reward}")

if __name__ == "__main__":
    test_tidybot_env()
