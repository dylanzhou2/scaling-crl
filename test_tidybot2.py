from brax.io import mjcf
from etils import epath

xml_path = epath.resource_path("brax") / "/Users/dylanzhou/Documents/Stanford/cs234/scaling-crl/envs/assets/stanford_tidybot2/tidybot.xml" # Or your absolute path
try:
    sys = mjcf.load(xml_path)
    print("Successfully loaded Tidybot MJCF!")
    print(f"Observation size: {sys.q_size()}, Action size: {sys.act_size()}")
except Exception as e:
    print(f"Brax loading failed: {e}")