import jax
from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy

def verify():
    print("Initializing TidyBotPushEasy...")
    try:
        env = TidyBotPushEasy(backend="mjx")
        print("\nInitialization Successful.")

        # Actuator count: 3 base + 7 arm + 1 gripper = 11.
        print(f"Actuator count (env.sys.nu): {env.sys.nu}")
        assert env.sys.nu == 11, (
            f"Expected 11 actuators (3 base + 7 arm + 1 gripper), got {env.sys.nu}. "
            "Action-conversion code in TidyBotEnv assumes 11; mismatch will cause silent "
            "wrong-actuator commands."
        )

        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        print(f"Observation Shape: {state.obs.shape} (Expected: (23,))")
        assert state.obs.shape == (23,), f"Expected obs shape (23,), got {state.obs.shape}"

        print("\nAll checks PASSED.")

    except Exception as e:
        print(f"\nFAILED: {e}")
        raise

if __name__ == "__main__":
    verify()