from brax.io import mjcf
from etils import epath

xml_path = epath.Path("/google/src/cloud/dylanzhou/scaling-crl/google3/video/youtube/enforcement/classifiers/ansible/experimental/dylan/scaling-crl/envs/assets/tidybot/stanford_tidybot/tidybot.xml")
try:
    sys = mjcf.load(xml_path)
    print("Brax actuator count:", sys.nu)
    print("Successfully loaded Tidybot MJCF!")
    print(f"Observation size: {sys.q_size()}, Action size: {sys.act_size()}")
except Exception as e:
    print(f"Brax loading failed: {e}")