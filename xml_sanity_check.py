import mujoco
import mujoco.viewer

model = mujoco.load_model_from_path("envs/assets/tidybot_push_easy.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
