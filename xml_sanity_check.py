"""Headless XML sanity check for tidybot_push_easy.

Loads the XML with the modern mujoco API, prints the actuator/joint/qpos
layout, then steps the sim under zero control to verify nothing explodes
or NaNs out. Useful for debugging mismatches between the XML and the
action-conversion code in TidyBotEnv.
"""

import numpy as np
import mujoco

XML_PATH = "envs/assets/tidybot_push_easy.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print(f"=== model summary for {XML_PATH} ===")
    print(f"nq (qpos size): {model.nq}")
    print(f"nv (qvel size): {model.nv}")
    print(f"nu (actuator count): {model.nu}")
    print(f"nbody: {model.nbody}, ngeom: {model.ngeom}, njnt: {model.njnt}")

    print("\n=== actuators (must match TidyBotEnv action-conversion order) ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "<unnamed>"
        ctrl_range = model.actuator_ctrlrange[i]
        print(f"  [{i:2d}] {name:30s} ctrl_range=[{ctrl_range[0]:+.3f}, {ctrl_range[1]:+.3f}]")

    print("\n=== joints ===")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "<unnamed>"
        jtype = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        qadr = model.jnt_qposadr[i]
        print(f"  [{i:2d}] {name:30s} type={jtype:5s} qpos_start={qadr}")

    print("\n=== zero-control stability check (200 steps) ===")
    mujoco.mj_resetData(model, data)
    qpos_initial = data.qpos.copy()
    for step in range(200):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            print(f"  NaN detected at step {step}. ABORT.")
            return
    qpos_final = data.qpos.copy()
    drift = float(np.linalg.norm(qpos_final - qpos_initial))
    print(f"  200 steps under zero ctrl: no NaN; qpos drift = {drift:.4f}")
    print(f"  cube qpos[18:25]: {qpos_final[18:25]}")
    print(f"  goal qpos[25:32]: {qpos_final[25:32]}")

    print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
