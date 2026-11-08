"""
Interactive Live 3D MuJoCo Viewer for Autonomous Laparoscopic Camera Control
============================================================================
Run this script locally on your laptop to open the interactive 3D MuJoCo window
and watch the 7-DOF Franka Emika Panda autonomously control the 30-degree
laparoscope inside the real Heidelberg OpenHELP CT human torso & organ system:

    c:\\Users\\arhaa\\Projects\\AgentVerse26\\prototypes\\.venv\\Scripts\\python.exe prototypes\\launch_live_viewer.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from franka_laparoscope_env import (
    build_scene_xml,
    solve_rcm_ik,
    set_instrument_from_rcm,
    PATIENT_ORIGIN,
    CAUTER_RCM_POS,
    GRASPER_RCM_POS,
)
from autonomous_laparoscope_agent import AutonomousLaparoscopeAgent


def main():
    xml = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Set initial home posture and solve initial RCM IK
    data.qpos[:7] = np.array([0.0, -0.35, 0.0, -1.75, 0.0, 1.55, 0.0])
    mujoco.mj_forward(model, data)

    rcm_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rcm_pivot_site")
    cauter_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cauter_tip_site")

    base_cauter_pos = PATIENT_ORIGIN + np.array([-0.042, 0.012, 0.102])
    base_grasper_pos = PATIENT_ORIGIN + np.array([-0.072, 0.018, 0.112])
    set_instrument_from_rcm(model, data, "surgical_cauter", CAUTER_RCM_POS, base_cauter_pos, roll_angle=0.2)
    set_instrument_from_rcm(model, data, "surgical_grasper", GRASPER_RCM_POS, base_grasper_pos, roll_angle=-0.3)
    mujoco.mj_forward(model, data)

    rcm_pos = data.site_xpos[rcm_site_id].copy()
    cauter_tip = data.site_xpos[cauter_site_id].copy()
    dir_init = (cauter_tip - rcm_pos) / np.linalg.norm(cauter_tip - rcm_pos)
    dist_init = np.linalg.norm(cauter_tip - rcm_pos)
    solve_rcm_ik(model, data, rcm_pos + dir_init * (dist_init - 0.090), rcm_pos, cauter_tip, max_iters=120)

    print("Scene compiled in <1s! Launching interactive 3D MuJoCo Viewer (60 FPS)...")
    print("Tip: In the viewer, press '[' or ']' to cycle between the 3rd-person OR view, Cavity Cutaway view, and 1st-person Endoscope Camera!")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_servo = -1.0
        while viewer.is_running():
            sim_time = data.time

            # Physiological breathing + active surgical dissection inside Calot's triangle
            resp_z = 0.0045 * np.sin(2.0 * np.pi * 0.25 * sim_time)
            resp_y = 0.0025 * np.cos(2.0 * np.pi * 0.25 * sim_time)
            data.ctrl[7] = resp_z
            data.ctrl[8] = resp_y

            dissect_offset = np.array([
                0.009 * np.sin(0.55 * sim_time) + 0.003 * np.cos(1.7 * sim_time),
                0.007 * np.cos(0.45 * sim_time),
                0.006 * np.sin(0.65 * sim_time) + resp_z,
            ])

            # Periodic unscripted surgical shift every 16 seconds
            cycle_t = sim_time % 16.0
            if 7.5 <= cycle_t <= 13.0:
                env_envelope = np.sin(np.pi * (cycle_t - 7.5) / 5.5) ** 2
                dissect_offset = dissect_offset + env_envelope * np.array([-0.015, 0.011, -0.009])

            set_instrument_from_rcm(
                model, data, "surgical_cauter", CAUTER_RCM_POS, base_cauter_pos + dissect_offset, roll_angle=0.2
            )
            set_instrument_from_rcm(
                model, data, "surgical_grasper", GRASPER_RCM_POS, base_grasper_pos + 0.65 * dissect_offset, roll_angle=-0.3
            )

            # Run 25 Hz RCM IK servoing directly on the main thread (avoids Windows WGL offscreen context contention)
            if sim_time - last_servo >= 0.04:
                rcm_p = data.site_xpos[rcm_site_id].copy()
                target_p = data.site_xpos[cauter_site_id].copy() + np.array([0.0, 0.005, -0.012])
                ray_v = target_p - rcm_p
                ray_d = np.linalg.norm(ray_v) + 1e-8
                ins_depth = np.clip(ray_d - 0.090, 0.045, 0.095)
                desired_tip = rcm_p + (ray_v / ray_d) * ins_depth
                solve_rcm_ik(model, data, desired_tip, rcm_p, target_p, max_iters=14)
                last_servo = sim_time

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.003)


if __name__ == "__main__":
    main()
