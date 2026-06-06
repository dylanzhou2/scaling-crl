from brax import envs
import jax

# Simulate what happens in tidybot_push_easy
try:
    from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy
    env = TidyBotPushEasy()
    print("Link names from XML:")
    print(env.sys.link_names)
    print("\nJoint names from XML:")
    print(env.sys.joint_names)
    print("\nActuator names from XML:")
    print(env.sys.actuator_names)
except Exception as e:
    import traceback
    traceback.print_exc()
