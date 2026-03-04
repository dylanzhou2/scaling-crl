import jax
from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy

def verify():
    print("Initializing TidyBotPushEasy...")
    # Instantiation triggers ArmEnvs.__init__ and TidyBotPushEasy.__init__
    try:
        env = TidyBotPushEasy(backend="mjx")
        print("\nInitialization Successful.")
        
        # Test a reset to ensure indices are valid
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        print(f"Observation Shape: {state.obs.shape} (Expected: 23)")
        
    except Exception as e:
        print(f"\nFAILED to initialize: {e}")

if __name__ == "__main__":
    verify()