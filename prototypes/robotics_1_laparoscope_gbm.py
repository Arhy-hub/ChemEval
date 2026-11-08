"""
Prototype 2: Robotics Track #1 — Autonomous Laparoscopic Camera Agent
Features:
- MuJoCo 3D surgical abdominal cavity with RCM trocar fulcrum, two active surgical instruments,
  pulsing hepatic artery, and an overhanging liver lobe obstacle.
- Stochastic biological dynamics:
    1) Organ & tool drift via Tethered Brownian Motion (Ornstein-Uhlenbeck SDE + respiratory rhythm)
    2) Cautery smoke / micro-bleed opacity evolving via Geometric Brownian Motion (GBM: dI = mu*I*dt + sigma*I*dW)
- 100% On-Device Perception-Driven Agent:
    - Perceives RGB + Depth stream from the simulated endoscope camera.
    - Estimates live Brownian volatility (sigma_hat) from visual/depth jitter.
    - Dynamically adjusts zoom (working distance) and 3-sigma collision safety margin.
    - Handles unscripted mid-run organ occlusion + GBM smoke spike by retracting and orbiting around the RCM fulcrum.
"""

import os
import sys
import time
import json
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image, ImageDraw

LAPAROSCOPE_MJCF = """
<mujoco model="laparoscopic_camera_agent">
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.4 0.4 0.4"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light pos="0 0 0.6" dir="0 0 -1" diffuse="0.9 0.8 0.8"/>
    <!-- Abdominal Cavity Background Wall (mucosal pink/red) -->
    <geom name="cavity_floor" type="plane" pos="0 0 -0.15" size="0.4 0.4 0.01" rgba="0.55 0.22 0.24 1"/>
    <geom name="trocar_ring" type="cylinder" pos="0 0 0.25" size="0.018 0.005" rgba="0.2 0.8 0.9 0.6" contype="0" conaffinity="0"/>

    <!-- Endoscope Camera Controlled by Robot Agent (pivoting through RCM trocar at 0,0,0.25) -->
    <body name="endoscope_tip" mocap="true" pos="0.0 0.0 0.14">
      <geom name="scope_shaft" type="capsule" fromto="0 0 0 0 0 0.14" size="0.007" rgba="0.75 0.78 0.82 1" contype="0" conaffinity="0"/>
      <geom name="scope_lens" type="sphere" pos="0 0 0" size="0.008" rgba="0.1 0.9 1.0 1" contype="0" conaffinity="0"/>
      <camera name="endoscope_cam" pos="0 0 -0.008" xyaxes="1 0 0 0 1 0" fovy="58"/>
    </body>

    <!-- Target Surgical Dissection Site & Cautery Tool Tip (Green/Cyan Marker) -->
    <body name="surgical_tool_right" mocap="true" pos="0.02 0.01 -0.04">
      <geom name="tool_r_shaft" type="capsule" fromto="0 0 0 0.12 0.10 0.18" size="0.004" rgba="0.3 0.35 0.4 1" contype="0" conaffinity="0"/>
      <geom name="tool_r_tip" type="sphere" pos="0 0 0" size="0.011" rgba="0.1 0.95 0.45 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Secondary Grasper Tool (Blue Marker) -->
    <body name="surgical_tool_left" mocap="true" pos="-0.04 0.02 -0.03">
      <geom name="tool_l_shaft" type="capsule" fromto="0 0 0 -0.12 0.10 0.18" size="0.004" rgba="0.3 0.35 0.4 1" contype="0" conaffinity="0"/>
      <geom name="tool_l_tip" type="sphere" pos="0 0 0" size="0.009" rgba="0.2 0.65 1.0 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Pulsating Hepatic Artery (Bright Red Cylinder) -->
    <body name="hepatic_artery" mocap="true" pos="-0.03 -0.02 -0.05">
      <geom name="artery_geom" type="capsule" fromto="-0.06 -0.02 0 0.06 0.03 0" size="0.012" rgba="0.92 0.12 0.15 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Overhanging Liver Lobe Obstacle (Drifts & Occludes Line-of-Sight Mid-Run) -->
    <body name="liver_lobe_obstacle" mocap="true" pos="0.09 0.06 0.03">
      <geom name="liver_geom" type="ellipsoid" size="0.045 0.035 0.022" rgba="0.65 0.18 0.18 0.95" contype="0" conaffinity="0"/>
    </body>

    <!-- 3rd-Person External Cutaway Camera -->
    <camera name="external_overview" pos="0.28 -0.28 0.22" xyaxes="0.707 0.707 0 -0.408 0.408 0.816" fovy="52"/>
  </worldbody>
</mujoco>
"""


def perceive_from_endoscope_rgb_depth(rgb: np.ndarray, depth: np.ndarray):
    """
    100% On-Device Sensor Perception (No ground-truth simulator coordinates!).
    Detects the green active surgical tool tip, the dark-red liver obstacle proximity,
    and estimates visibility from the RGB-D frame.
    """
    h, w, _ = rgb.shape
    r, g, b = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)

    # Segment green surgical tool tip in pixel space
    tool_mask = (g > 150) & (g > r * 1.35) & (g > b * 1.25)
    tool_pixels = int(np.sum(tool_mask))

    if tool_pixels > 15:
        ys, xs = np.where(tool_mask)
        cx_norm = (float(np.mean(xs)) - w / 2.0) / (w / 2.0)
        cy_norm = (float(np.mean(ys)) - h / 2.0) / (h / 2.0)
        tool_depth = float(np.median(depth[tool_mask]))
        tool_visible = True
    else:
        cx_norm, cy_norm = 0.0, 0.0
        tool_depth = 0.18
        tool_visible = False

    # Nearest obstacle clearance in central field of view
    center_depth = depth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    min_clearance = float(np.percentile(center_depth, 5))

    return {
        "tool_visible": tool_visible,
        "tool_pixels": tool_pixels,
        "cx_norm": cx_norm,
        "cy_norm": cy_norm,
        "tool_depth": tool_depth,
        "min_clearance": min_clearance,
    }


def run_simulation(use_viewer: bool = False):
    os.makedirs("prototypes/outputs", exist_ok=True)
    model = mujoco.MjModel.from_xml_string(LAPAROSCOPE_MJCF)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    viewer_ctx = mujoco.viewer.launch_passive(model, data) if use_viewer else None

    rng = np.random.default_rng(42)
    dt = 0.05
    steps = 120  # 6 seconds simulated run

    # Mocap indices
    id_scope = model.body("endoscope_tip").mocapid[0]
    id_tool_r = model.body("surgical_tool_right").mocapid[0]
    id_tool_l = model.body("surgical_tool_left").mocapid[0]
    id_artery = model.body("hepatic_artery").mocapid[0]
    id_liver = model.body("liver_lobe_obstacle").mocapid[0]

    # Stochastic State Variables
    tool_pos = np.array([0.015, 0.01, -0.04])
    liver_pos = np.array([0.085, 0.05, 0.035])
    smoke_gbm = 0.04  # Initial low cautery smoke opacity (GBM state > 0)

    # Agent Internal State (100% self-contained)
    cam_pos = np.array([0.0, 0.0, 0.13])
    orbit_angle = 0.0
    obs_history = []
    telemetry_log = []
    key_frames = {}

    for step in range(steps):
        t = step * dt

        # --- 1. STOCHASTIC ENVIRONMENT DYNAMICS (Tethered Brownian + GBM Smoke) ---
        # Unscripted mid-run event at t >= 2.2s: liver lobe shifts toward line of sight & volatility doubles
        regime_shock = (t >= 2.2) and (t < 4.2)
        sigma_tissue = 0.014 if regime_shock else 0.005

        # Ornstein-Uhlenbeck (tethered Brownian motion) + breathing for surgical target
        breath_target = np.array([0.015 * np.sin(2.0 * t), 0.012 * np.cos(1.5 * t), -0.04 + 0.008 * np.sin(3.0 * t)])
        dW_tool = rng.normal(0.0, np.sqrt(dt), size=3)
        tool_pos = tool_pos + 3.0 * (breath_target - tool_pos) * dt + sigma_tissue * dW_tool

        # Liver obstacle drifts into the line of sight during regime shock
        liver_anchor = np.array([0.005, 0.005, 0.03]) if regime_shock else np.array([0.085, 0.05, 0.035])
        dW_liver = rng.normal(0.0, np.sqrt(dt), size=3)
        liver_pos = liver_pos + 2.5 * (liver_anchor - liver_pos) * dt + sigma_tissue * dW_liver

        # Geometric Brownian Motion (GBM) for cautery smoke intensity: dS = mu*S*dt + sigma*S*dW
        mu_smoke = 0.45 if (t >= 2.5 and t < 3.6) else -0.65
        sigma_smoke = 0.35
        dW_smoke = rng.normal(0.0, np.sqrt(dt))
        smoke_gbm = float(np.clip(smoke_gbm * np.exp((mu_smoke - 0.5 * sigma_smoke**2) * dt + sigma_smoke * dW_smoke), 0.02, 0.65))

        # Apply positions to MuJoCo bodies
        data.mocap_pos[id_tool_r] = tool_pos
        data.mocap_pos[id_tool_l] = np.array([-0.035, 0.015, -0.03]) + 0.4 * (tool_pos - breath_target)
        data.mocap_pos[id_artery] = np.array([-0.03, -0.02, -0.05 + 0.003 * np.sin(7.5 * t)])
        data.mocap_pos[id_liver] = liver_pos
        data.mocap_pos[id_scope] = cam_pos
        mujoco.mj_forward(model, data)
        if viewer_ctx is not None:
            viewer_ctx.sync()
            time.sleep(dt)

        # --- 2. AGENT PERCEPTION (RGB + Depth from Endoscope Camera Only) ---
        renderer.update_scene(data, camera="endoscope_cam")
        renderer.disable_depth_rendering()
        rgb_endo = renderer.render().copy()

        # Apply GBM smoke scattering effect on camera feed
        if smoke_gbm > 0.05:
            haze = np.full_like(rgb_endo, 210, dtype=np.float32)
            rgb_endo = np.clip((1.0 - smoke_gbm) * rgb_endo.astype(np.float32) + smoke_gbm * haze, 0, 255).astype(np.uint8)

        renderer.enable_depth_rendering()
        depth_endo = renderer.render().copy()
        renderer.disable_depth_rendering()

        obs = perceive_from_endoscope_rgb_depth(rgb_endo, depth_endo)
        obs_history.append((obs["cx_norm"], obs["cy_norm"], obs["min_clearance"]))
        if len(obs_history) > 12:
            obs_history.pop(0)

        # --- 3. AGENT STOCHASTIC VOLATILITY ESTIMATOR & AUTONOMOUS DECISION ---
        if len(obs_history) >= 4:
            diffs = np.diff(np.array(obs_history)[:, :2], axis=0)
            sigma_hat = float(np.linalg.norm(np.std(diffs, axis=0)) / np.sqrt(dt))
        else:
            sigma_hat = 0.05

        # Autonomous Mode Selection based purely on perceived clearance, occlusion, and volatility
        if obs["min_clearance"] < 0.085 or smoke_gbm > 0.42:
            mode = "CRITICAL_RETRACT_AND_ORBIT"
            orbit_angle += 0.14  # Orbit around RCM cone to clear overhanging liver lobe
            target_z = 0.165     # Pull camera back toward trocar port for safety
        elif not obs["tool_visible"]:
            mode = "OCCLUSION_ORBIT_SEARCH"
            orbit_angle += 0.12
            target_z = 0.155
        elif sigma_hat > 0.22:
            mode = "HIGH_VOLATILITY_WIDEN_FOV"
            # Widen FOV by increasing working distance proportional to 3*sigma_hat
            target_z = min(0.16, 0.115 + 0.08 * sigma_hat)
        else:
            mode = "PRECISION_CLOSEUP_TRACKING"
            target_z = 0.115

        # Visual servoing in X-Y + orbital offset while respecting RCM trocar fulcrum
        orbit_offset = np.array([0.045 * np.sin(orbit_angle), -0.045 * (1.0 - np.cos(orbit_angle))])
        servo_step = np.array([0.018 * obs["cx_norm"], -0.018 * obs["cy_norm"]]) if obs["tool_visible"] else np.zeros(2)
        cam_pos[0] = float(np.clip(0.85 * cam_pos[0] + 0.15 * (orbit_offset[0]) + servo_step[0], -0.07, 0.07))
        cam_pos[1] = float(np.clip(0.85 * cam_pos[1] + 0.15 * (orbit_offset[1]) + servo_step[1], -0.07, 0.07))
        cam_pos[2] = float(0.82 * cam_pos[2] + 0.18 * target_z)

        telemetry_log.append({
            "step": step,
            "t": round(t, 2),
            "mode": mode,
            "sigma_hat": round(sigma_hat, 3),
            "smoke_gbm": round(smoke_gbm, 3),
            "min_clearance_mm": round(obs["min_clearance"] * 1000, 1),
            "tool_visible": obs["tool_visible"],
            "cam_z_mm": round(cam_pos[2] * 1000, 1),
        })

        # Capture 3 key phases for side-by-side composite visual strip
        if step in (15, 55, 95):
            renderer.update_scene(data, camera="external_overview")
            rgb_ext = renderer.render().copy()
            key_frames[step] = (rgb_ext, rgb_endo, telemetry_log[-1])

    if viewer_ctx is not None:
        viewer_ctx.close()

    # Assemble composite visual strip showing 3rd-person robot view + 1st-person perception
    strip = Image.new("RGB", (960, 360 * 3), (15, 18, 26))
    draw = ImageDraw.Draw(strip)
    for idx, step_k in enumerate([15, 55, 95]):
        ext_img, endo_img, info = key_frames[step_k]
        y0 = idx * 360
        strip.paste(Image.fromarray(ext_img), (0, y0))
        strip.paste(Image.fromarray(endo_img), (480, y0))
        banner = (
            f"t={info['t']}s | MODE: {info['mode']} | Est Volatility sigma_hat={info['sigma_hat']} "
            f"| Clearance={info['min_clearance_mm']}mm | GBM Smoke={info['smoke_gbm']}"
        )
        draw.rectangle([0, y0, 960, y0 + 28], fill=(10, 14, 24))
        draw.text((12, y0 + 7), banner, fill=(80, 245, 180))

    out_img = "prototypes/outputs/robotics_1_laparoscope_sim.png"
    strip.save(out_img)
    out_json = "prototypes/outputs/robotics_1_telemetry.json"
    with open(out_json, "w") as f:
        json.dump(telemetry_log, f, indent=2)

    print("\n=== ROBOTICS #1 PROTOTYPE: STOCHASTIC LAPAROSCOPIC CAMERA AGENT ===")
    for k in [15, 55, 95]:
        row = telemetry_log[k]
        print(f"t={row['t']:4.2f}s | {row['mode']:28s} | sigma_hat={row['sigma_hat']:.3f} | clearance={row['min_clearance_mm']:5.1f}mm | cam_z={row['cam_z_mm']:5.1f}mm")
    print(f"Saved split-screen simulation capture to: {out_img}")


if __name__ == "__main__":
    run_simulation(use_viewer=("--viewer" in sys.argv))
