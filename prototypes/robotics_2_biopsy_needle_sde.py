"""
Prototype 3: Robotics Track #2 — Autonomous Biopsy Needle Agent with Stochastic Brownian Drift & GBM Volatility
Features:
- MuJoCo 3D surgical scene: robotic biopsy needle, ribcage bars, pulsating hepatic artery (danger zone),
  and moving liver lesion (target).
- How Brownian Motion & GBM are used here:
    1) Target tumor & artery move with Quasi-Periodic Breathing + Tethered Brownian Motion (dW_t).
    2) Physiological Volatility (sigma_t) evolves via Geometric Brownian Motion (GBM) with a sudden
       Jump-Diffusion regime spike at t=2.2s (simulating an unscripted patient cough & permanent arterial shift).
- 100% On-Device Perception-Driven Agent:
    - Perceives noisy 3D depth/camera measurements + simulated wrist lateral shear force.
    - Runs an online Kalman-Bucy / SDE state estimator tracking position, velocity, and 3-sigma uncertainty radius.
    - Refuses to advance when 3-sigma uncertainty > safety corridor margin ("behaves sensibly when unsure").
    - Triggers instant axial emergency retraction on force/volatility spike, then re-plans a new entry angle
      around the shifted hepatic artery.
"""

import os
import sys
import time
import json
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image, ImageDraw

BIOPSY_MJCF = """
<mujoco model="biopsy_needle_sde_agent">
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35" specular="0.4 0.4 0.4"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light pos="0 -0.2 0.55" dir="0 0.3 -1" diffuse="0.9 0.85 0.85"/>
    <geom name="table_bed" type="plane" pos="0 0 -0.14" size="0.4 0.4 0.01" rgba="0.18 0.22 0.28 1"/>

    <!-- Translucent Soft Tissue Liver Phantom -->
    <body name="liver_phantom" mocap="true" pos="0 0 -0.04">
      <geom name="liver_volume" type="ellipsoid" size="0.11 0.08 0.045" rgba="0.68 0.24 0.26 0.38" contype="0" conaffinity="0"/>
    </body>

    <!-- Ribcage Bone Obstacles -->
    <geom name="rib_8" type="capsule" fromto="-0.09 -0.045 0.015 0.09 -0.045 0.015" size="0.008" rgba="0.92 0.90 0.82 0.9" contype="0" conaffinity="0"/>
    <geom name="rib_9" type="capsule" fromto="-0.09 0.045 0.015 0.09 0.045 0.015" size="0.008" rgba="0.92 0.90 0.82 0.9" contype="0" conaffinity="0"/>

    <!-- Target Tumor Lesion (Bright Green Sphere inside Liver) -->
    <body name="tumor_target" mocap="true" pos="0.0 0.0 -0.045">
      <geom name="tumor_geom" type="sphere" size="0.013" rgba="0.15 0.95 0.40 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Hepatic Artery Danger Zone (Bright Red Pulsating Vessel near Corridor) -->
    <body name="hepatic_artery" mocap="true" pos="0.018 0.0 -0.015">
      <geom name="artery_vessel" type="capsule" fromto="0 -0.065 0 0 0.065 0" size="0.010" rgba="0.95 0.12 0.15 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Robotic Biopsy Needle Holder & Shaft -->
    <body name="needle_tip" mocap="true" pos="0.0 0.0 0.09">
      <geom name="needle_shaft" type="capsule" fromto="0 0 0 0 0 0.11" size="0.003" rgba="0.82 0.88 0.95 1" contype="0" conaffinity="0"/>
      <geom name="needle_hub" type="cylinder" pos="0 0 0.11" size="0.012 0.018" rgba="0.2 0.75 0.95 1" contype="0" conaffinity="0"/>
      <geom name="tip_marker" type="sphere" pos="0 0 0" size="0.0045" rgba="0.1 0.95 1.0 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Cameras: Side Depth/Ultrasound Slice View & 3D Isometric Overview -->
    <camera name="iso_view" pos="0.0 -0.27 0.13" xyaxes="1 0 0 0 0.42 0.91" fovy="46"/>
    <camera name="top_sensor_cam" pos="0.24 0.0 0.14" xyaxes="0 1 0 -0.48 0 0.88" fovy="46"/>
  </worldbody>
</mujoco>
"""


def euler_to_quat_y(tilt_rad: float) -> np.ndarray:
    """Quaternion rotation around Y-axis for tilting needle corridor angle."""
    return np.array([np.cos(tilt_rad / 2.0), 0.0, np.sin(tilt_rad / 2.0), 0.0])


def run_biopsy_simulation(use_viewer: bool = False):
    os.makedirs("prototypes/outputs", exist_ok=True)
    model = mujoco.MjModel.from_xml_string(BIOPSY_MJCF)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    viewer_ctx = mujoco.viewer.launch_passive(model, data) if use_viewer else None

    rng = np.random.default_rng(7)
    dt = 0.05
    steps = 120

    id_liver = model.body("liver_phantom").mocapid[0]
    id_tumor = model.body("tumor_target").mocapid[0]
    id_artery = model.body("hepatic_artery").mocapid[0]
    id_needle = model.body("needle_tip").mocapid[0]

    # Stochastic Biology State
    tumor_pos = np.array([0.0, 0.0, -0.045])
    artery_offset_x = 0.022  # Starts safely 22mm to the right of vertical corridor
    sigma_gbm = 0.003        # Initial low Brownian volatility (3mm/sqrt(s))

    # Agent Internal State (Kalman-Bucy Filter + Corridor Planner)
    est_tumor = np.array([0.0, 0.0, -0.045])
    est_uncertainty_mm = 4.5  # Starts high while learning breathing phase
    needle_pos = np.array([0.0, 0.0, 0.085])
    corridor_tilt_rad = 0.0   # Starts vertical (Corridor A: 0 deg)
    inserted_depth = 0.0
    replanned_after_cough = False

    telemetry_log = []
    key_frames = {}

    for step in range(steps):
        t = step * dt

        # --- 1. STOCHASTIC PHYSIOLOGY (Breathing + Brownian Drift + GBM Volatility + Cough Jump) ---
        cough_event = (step >= 45) and (step <= 54)
        mu_vol = 2.8 if cough_event else -1.4
        sig_vol = 0.45
        dW_vol = rng.normal(0.0, np.sqrt(dt))
        sigma_gbm = float(np.clip(sigma_gbm * np.exp((mu_vol - 0.5 * sig_vol**2) * dt + sig_vol * dW_vol), 0.0025, 0.028))

        # At cough onset (step 46), hepatic artery permanently shifts left into vertical Corridor A!
        if step == 46:
            artery_offset_x = 0.002  # Now directly blocks the 0-degree vertical path!

        # Quasi-periodic respiratory wave + Tethered Brownian Motion dW
        breath_anchor = np.array([
            0.008 * np.sin(2.2 * t),
            0.0,
            -0.045 + 0.012 * np.sin(2.2 * t) + 0.002 * np.sin(8.5 * t),
        ])
        cough_impulse = np.array([-0.022, 0.0, 0.025]) if (step == 46) else np.zeros(3)
        dW_tissue = rng.normal(0.0, np.sqrt(dt), size=3)
        tumor_pos = tumor_pos + 4.0 * (breath_anchor - tumor_pos) * dt + sigma_gbm * dW_tissue + cough_impulse
        artery_pos = tumor_pos + np.array([artery_offset_x, 0.0, 0.028])

        # --- 2. AGENT SENSOR PERCEPTION (Noisy Depth Camera + Wrist Lateral Shear Force) ---
        sensor_noise = rng.normal(0.0, 0.0012, size=3)
        perceived_tumor = tumor_pos + sensor_noise
        perceived_artery = artery_pos + rng.normal(0.0, 0.0012, size=3)

        # Wrist force sensor feels lateral shear if needle is inside tissue during a cough jolt
        in_tissue = needle_pos[2] < 0.005
        pred_error_mm = float(np.linalg.norm(perceived_tumor - est_tumor) * 1000.0)
        wrist_shear_force_N = (0.85 * pred_error_mm) if in_tissue else 0.15

        # Update Agent's On-Device Kalman-Bucy State & Uncertainty Estimate
        kalman_gain = 0.55
        est_tumor = est_tumor + kalman_gain * (perceived_tumor - est_tumor)
        target_unc = 0.45 * pred_error_mm + 95.0 * sigma_gbm
        est_uncertainty_mm = float(np.clip(0.55 * est_uncertainty_mm + 0.45 * target_unc, 0.8, 14.0))

        # --- 3. AGENT AUTONOMOUS DECISION & CORRIDOR RE-PLANNING ---
        if cough_event or wrist_shear_force_N > 3.5 or est_uncertainty_mm > 3.2:
            mode = "EMERGENCY_AXIAL_RETRACT"
            inserted_depth = max(0.0, inserted_depth - 0.045)
            if step >= 45:
                replanned_after_cough = True
        elif replanned_after_cough and abs(corridor_tilt_rad) < 0.1:
            # Check if Corridor A (0 deg) is blocked by shifted artery
            artery_clearance_corridor_a = abs(perceived_artery[0] - est_tumor[0]) * 1000.0
            if artery_clearance_corridor_a < 10.0:
                mode = "REPLAN_CORRIDOR_B_ANGLED"
                corridor_tilt_rad = -0.42  # Tilt needle -24 degrees to bypass shifted artery on the left!
        elif est_uncertainty_mm > 2.2:
            mode = "UNCERTAINTY_GATE_HOVER_WAIT"
            # Wait hovering safely outside tissue until Brownian/breathing tracking locks in
            inserted_depth = max(0.0, inserted_depth - 0.01)
        else:
            mode = "PHASE_LOCKED_INSERTION"
            inserted_depth = min(0.115, inserted_depth + 0.012)

        # Compute 3D needle tip position along chosen corridor angle synchronized with estimated tumor phase
        corridor_dir = np.array([np.sin(corridor_tilt_rad), 0.0, -np.cos(corridor_tilt_rad)])
        hover_entry_point = est_tumor - corridor_dir * 0.115
        needle_pos = hover_entry_point + corridor_dir * inserted_depth

        # Update MuJoCo scene
        data.mocap_pos[id_liver] = np.array([tumor_pos[0], 0.0, tumor_pos[2] + 0.005])
        data.mocap_pos[id_tumor] = tumor_pos
        data.mocap_pos[id_artery] = artery_pos
        data.mocap_pos[id_needle] = needle_pos
        data.mocap_quat[id_needle] = euler_to_quat_y(corridor_tilt_rad)
        mujoco.mj_forward(model, data)
        if viewer_ctx is not None:
            viewer_ctx.sync()
            time.sleep(dt)

        dist_to_target_mm = float(np.linalg.norm(needle_pos - tumor_pos) * 1000.0)
        telemetry_log.append({
            "step": step,
            "t": round(t, 2),
            "mode": mode,
            "uncertainty_3sigma_mm": round(est_uncertainty_mm, 2),
            "wrist_force_N": round(wrist_shear_force_N, 2),
            "corridor_angle_deg": round(np.degrees(corridor_tilt_rad), 1),
            "dist_to_tumor_mm": round(dist_to_target_mm, 1),
        })

        if step in (30, 47, 108):
            renderer.update_scene(data, camera="iso_view")
            rgb_iso = renderer.render().copy()
            renderer.update_scene(data, camera="top_sensor_cam")
            rgb_side = renderer.render().copy()
            key_frames[step] = (rgb_iso, rgb_side, telemetry_log[-1])

    if viewer_ctx is not None:
        viewer_ctx.close()

    # Save visual composite strip
    strip = Image.new("RGB", (960, 360 * 3), (15, 18, 26))
    draw = ImageDraw.Draw(strip)
    for idx, step_k in enumerate([30, 47, 108]):
        iso_img, side_img, info = key_frames[step_k]
        y0 = idx * 360
        strip.paste(Image.fromarray(iso_img), (0, y0))
        strip.paste(Image.fromarray(side_img), (480, y0))
        banner = (
            f"t={info['t']}s | {info['mode']} | 3-Sigma Uncertainty={info['uncertainty_3sigma_mm']}mm "
            f"| Force={info['wrist_force_N']}N | Angle={info['corridor_angle_deg']}deg | Dist={info['dist_to_tumor_mm']}mm"
        )
        draw.rectangle([0, y0, 960, y0 + 28], fill=(10, 14, 24))
        draw.text((12, y0 + 7), banner, fill=(255, 215, 90) if step_k == 47 else (80, 245, 180))

    out_img = "prototypes/outputs/robotics_2_biopsy_sim.png"
    strip.save(out_img)
    out_json = "prototypes/outputs/robotics_2_telemetry.json"
    with open(out_json, "w") as f:
        json.dump(telemetry_log, f, indent=2)

    print("\n=== ROBOTICS #2 PROTOTYPE: STOCHASTIC BIOPSY NEEDLE AGENT ===")
    for k in [30, 47, 108]:
        row = telemetry_log[k]
        print(
            f"t={row['t']:4.2f}s | {row['mode']:26s} | 3sigma={row['uncertainty_3sigma_mm']:4.2f}mm "
            f"| Force={row['wrist_force_N']:4.2f}N | Angle={row['corridor_angle_deg']:5.1f}deg | Dist={row['dist_to_tumor_mm']:4.1f}mm"
        )
    print(f"Saved split-screen simulation capture to: {out_img}")


if __name__ == "__main__":
    run_biopsy_simulation(use_viewer=("--viewer" in sys.argv))
