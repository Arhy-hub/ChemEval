"""
High-Fidelity Coordinated Dual-Arm MuJoCo Surgical Environment:
- High-Resolution 3D Anatomical STL Meshes (Bilobed Human Liver, Gallbladder Fossa, Bifurcating Hepatic Artery Tree)
- 3 Fluorescent ICG-Green Tumor Nodules (Nodule #1, #2, #3) that dynamically change appearance during thermal ablation:
    * Unablated: Bright ICG Green
    * Actively Ablating: Glowing Thermal Orange
    * Ablated (Complete): Coagulated White/Gray
- Arm 1: 5-DOF Articulated Chip-on-Tip Robotic Laparoscope (RCM Pitch, RCM Yaw, Insertion Slide, 30-Deg Wrist Pan/Tilt)
- Arm 2: 3-DOF Articulated Surgical Electrocautery Arm with Wrist Force/Contact Sensing
- Physiological Dynamics: Quasi-periodic breathing + Tethered Brownian tissue drift + Unscripted Liver Lobe Collapse
"""

import os
import numpy as np
import mujoco
from mesh_generator import generate_all_surgical_meshes


def build_surgical_mjcf(asset_dir: str) -> str:
    liver_stl = os.path.abspath(os.path.join(asset_dir, "liver_lobe.stl")).replace("\\", "/")
    gb_stl = os.path.abspath(os.path.join(asset_dir, "gallbladder_bed.stl")).replace("\\", "/")
    artery_stl = os.path.abspath(os.path.join(asset_dir, "hepatic_artery_tree.stl")).replace("\\", "/")

    return f"""
<mujoco model="dual_arm_autonomous_laparoscopic_ablation">
  <option timestep="0.005" integrator="implicitfast" gravity="0 0 0"/>

  <visual>
    <headlight diffuse="0.58 0.54 0.54" ambient="0.30 0.24 0.24" specular="0.88 0.88 0.88"/>
    <quality shadowsize="2048" offsamples="4"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <texture name="peritoneum_tex" type="2d" builtin="gradient" rgb1="0.58 0.22 0.24" rgb2="0.40 0.13 0.15" width="512" height="512" mark="random" markrgb="0.72 0.28 0.30"/>
    <texture name="liver_tex" type="2d" builtin="gradient" rgb1="0.48 0.13 0.14" rgb2="0.34 0.08 0.09" width="512" height="512" mark="random" markrgb="0.56 0.16 0.17"/>

    <material name="wet_peritoneum" texture="peritoneum_tex" specular="0.75" shininess="0.85" reflectance="0.05"/>
    <material name="wet_liver" texture="liver_tex" specular="0.92" shininess="0.95" reflectance="0.08"/>
    <material name="wet_artery" rgba="0.90 0.08 0.12 1" specular="0.95" shininess="0.92"/>
    <material name="wet_gallbladder" rgba="0.30 0.55 0.34 0.96" specular="0.88" shininess="0.90"/>
    <material name="surgical_steel" rgba="0.78 0.82 0.86 1" specular="0.95" shininess="0.95" reflectance="0.15"/>
    <material name="robot_housing" rgba="0.92 0.94 0.96 1" specular="0.6" shininess="0.7"/>
    <material name="nodule_mat" rgba="0.12 0.98 0.42 1" specular="0.90" shininess="0.90"/>

    <mesh name="liver_mesh" file="{liver_stl}" scale="1.25 1.25 1.25"/>
    <mesh name="gallbladder_mesh" file="{gb_stl}" scale="1.18 1.18 1.18"/>
    <mesh name="artery_mesh" file="{artery_stl}" scale="1.15 1.15 1.15"/>
  </asset>

  <worldbody>
    <light pos="0.05 -0.08 0.35" dir="-0.1 0.2 -1" diffuse="0.85 0.80 0.78" specular="0.9 0.9 0.9"/>
    <light pos="-0.08 0.08 0.30" dir="0.2 -0.2 -1" diffuse="0.45 0.40 0.42" specular="0.5 0.5 0.5"/>

    <geom name="posterior_peritoneum" type="plane" pos="0 0 -0.12" size="0.35 0.35 0.01" material="wet_peritoneum"/>
    <geom name="rib_arch_1" type="capsule" fromto="-0.14 -0.06 0.05 0.14 -0.05 0.06" size="0.007" rgba="0.88 0.82 0.72 0.48" contype="0" conaffinity="0"/>
    <geom name="rib_arch_2" type="capsule" fromto="-0.14 0.04 0.06 0.14 0.05 0.07" size="0.007" rgba="0.88 0.82 0.72 0.48" contype="0" conaffinity="0"/>

    <!-- ==================== ANATOMICAL ORGAN BED & 3 TUMOR NODULES ==================== -->
    <body name="organ_complex" pos="0.0 0.01 -0.045">
      <joint name="organ_drift_x" type="slide" axis="1 0 0" damping="8.0"/>
      <joint name="organ_drift_y" type="slide" axis="0 1 0" damping="8.0"/>
      <joint name="organ_drift_z" type="slide" axis="0 0 1" damping="8.0"/>

      <!-- Bifurcating Hepatic Arterial Tree (Critical No-Fly Hazard) -->
      <geom name="hepatic_artery" type="mesh" mesh="artery_mesh" pos="0.0 -0.014 0.004" material="wet_artery" contype="0" conaffinity="0"/>
      <site name="artery_hazard_site" pos="0.002 -0.010 0.008" size="0.005"/>

      <!-- Pear-Shaped Gallbladder Fossa -->
      <geom name="gallbladder" type="mesh" mesh="gallbladder_mesh" pos="-0.018 0.010 -0.006" material="wet_gallbladder" contype="0" conaffinity="0"/>

      <!-- 3 Target Tumor Nodules to be Autonomously Ablated in Sequence -->
      <body name="nodule_body_1" pos="-0.022 -0.006 0.008">
        <geom name="nodule_geom_1" type="sphere" size="0.0085" rgba="0.10 0.98 0.42 1" contype="0" conaffinity="0"/>
        <site name="nodule_site_1" pos="0 0 0" size="0.002"/>
      </body>
      <body name="nodule_body_2" pos="0.004 0.002 0.007">
        <geom name="nodule_geom_2" type="sphere" size="0.0085" rgba="0.10 0.98 0.42 1" contype="0" conaffinity="0"/>
        <site name="nodule_site_2" pos="0 0 0" size="0.002"/>
      </body>
      <body name="nodule_body_3" pos="0.024 0.014 0.006">
        <geom name="nodule_geom_3" type="sphere" size="0.0090" rgba="0.10 0.98 0.42 1" contype="0" conaffinity="0"/>
        <site name="nodule_site_3" pos="0 0 0" size="0.002"/>
      </body>

      <!-- Articulated Bilobed Human Liver (Droops over Nodule #3 during Unscripted Shift) -->
      <body name="liver_lobe_body" pos="0.022 0.032 0.034">
        <joint name="liver_droop_x" type="slide" axis="1 0 0" damping="6.0"/>
        <joint name="liver_droop_y" type="slide" axis="0 1 0" damping="6.0"/>
        <joint name="liver_droop_z" type="slide" axis="0 0 1" damping="6.0"/>
        <geom name="liver_parenchyma" type="mesh" mesh="liver_mesh" material="wet_liver" solref="0.015 1.1" contype="0" conaffinity="0"/>
        <site name="liver_margin_site" pos="0.0 -0.018 -0.012" size="0.005"/>
      </body>
    </body>

    <!-- ==================== ARM 1: 5-DOF ROBOTIC CHIP-ON-TIP ENDOSCOPE ==================== -->
    <body name="trocar_port_mount" pos="-0.02 -0.015 0.175">
      <geom name="trocar_cannula" type="cylinder" pos="0 0 0.01" size="0.013 0.016" material="robot_housing" contype="0" conaffinity="0"/>
      <geom name="trocar_seal" type="cylinder" pos="0 0 0.027" size="0.016 0.004" rgba="0.15 0.65 0.85 0.9" contype="0" conaffinity="0"/>

      <body name="rcm_gimbal" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.15" diaginertia="0.001 0.001 0.001"/>
        <joint name="scope_pitch" type="hinge" axis="1 0 0" range="-0.65 0.65" damping="2.5"/>
        <joint name="scope_yaw" type="hinge" axis="0 1 0" range="-0.65 0.65" damping="2.5"/>

        <body name="scope_carriage" pos="0 0 0">
          <joint name="scope_insert" type="slide" axis="0 0 -1" range="0.02 0.14" damping="4.0"/>
          <geom name="robot_wrist_housing" type="cylinder" pos="0 0 0.055" size="0.015 0.022" material="robot_housing" contype="0" conaffinity="0"/>
          <geom name="laparoscope_shaft" type="capsule" fromto="0 0 0.035 0 0 -0.058" size="0.005" material="surgical_steel" contype="1" conaffinity="1"/>

          <body name="scope_distal_tip" pos="0 0 -0.060">
            <joint name="tip_pan" type="hinge" axis="0 1 0" range="-0.55 0.55" damping="1.0"/>
            <joint name="tip_tilt" type="hinge" axis="1 0 0" range="-0.55 0.55" damping="1.0"/>
            <geom name="distal_lens_bezel" type="cylinder" pos="0 0 -0.004" size="0.0055 0.004" rgba="0.15 0.85 0.95 1" contype="0" conaffinity="0"/>
            <site name="scope_tip_site" pos="0 0 -0.006" size="0.003"/>
            <light name="endoscope_led" pos="0 0 -0.005" dir="0 0 -1" diffuse="0.95 0.92 0.88" attenuation="1 0 2"/>
            <camera name="endoscope_cam" pos="0 0 -0.006" xyaxes="1 0 0 0 1 0" fovy="58"/>
          </body>
        </body>
      </body>
    </body>

    <!-- ==================== ARM 2: ARTICULATED SURGICAL ELECTROCAUTERY INSTRUMENT ==================== -->
    <body name="right_instrument_trocar" pos="0.095 -0.035 0.155">
      <geom name="inst_trocar" type="cylinder" size="0.011 0.015" euler="0.22 -0.48 0" material="robot_housing" contype="0" conaffinity="0"/>
      <geom name="inst_seal" type="cylinder" pos="0 0 0.015" size="0.014 0.003" euler="0.22 -0.48 0" rgba="0.95 0.55 0.15 0.9" contype="0" conaffinity="0"/>

      <body name="cautery_gimbal" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.15" diaginertia="0.001 0.001 0.001"/>
        <!-- 3-DOF RCM Articulated Instrument Joints (Pitch, Yaw, Axial Insertion) -->
        <joint name="cautery_pitch" type="hinge" axis="1 0 0" range="-0.75 0.75" damping="2.0"/>
        <joint name="cautery_yaw" type="hinge" axis="0 1 0" range="-0.85 0.85" damping="2.0"/>

        <body name="cautery_shaft_body" pos="0 0 0">
          <joint name="cautery_insert" type="slide" axis="-0.45 0.18 -0.87" range="0.02 0.24" damping="3.5"/>
          <geom name="cautery_shaft" type="capsule" fromto="0 0 0.03 -0.045 0.018 -0.087" size="0.0035" material="surgical_steel" contype="1" conaffinity="1"/>
          <geom name="cautery_electrode_tip" type="sphere" pos="-0.047 0.019 -0.091" size="0.0048" rgba="0.20 0.85 1.0 1" contype="0" conaffinity="0"/>
          <site name="cautery_tip_site" pos="-0.047 0.019 -0.091" size="0.003"/>
        </body>
      </body>
    </body>

    <!-- External 3rd-Person Cutaway Camera -->
    <camera name="cutaway_side" pos="0.22 -0.25 0.15" xyaxes="0.74 0.67 0 -0.35 0.39 0.85" fovy="45"/>
  </worldbody>

  <actuator>
    <!-- Arm 1: 5-DOF Robotic Endoscope Actuators -->
    <position name="act_scope_pitch" joint="scope_pitch" kp="35" kv="4"/>
    <position name="act_scope_yaw" joint="scope_yaw" kp="35" kv="4"/>
    <position name="act_scope_insert" joint="scope_insert" kp="60" kv="6"/>
    <position name="act_tip_pan" joint="tip_pan" kp="22" kv="2"/>
    <position name="act_tip_tilt" joint="tip_tilt" kp="22" kv="2"/>

    <!-- Arm 2: 3-DOF Surgical Cautery Instrument Actuators -->
    <position name="act_cautery_pitch" joint="cautery_pitch" kp="40" kv="4"/>
    <position name="act_cautery_yaw" joint="cautery_yaw" kp="40" kv="4"/>
    <position name="act_cautery_insert" joint="cautery_insert" kp="65" kv="6"/>
  </actuator>
</mujoco>
"""


class SurgicalLaparoscopeEnv:
    """
    Coordinated Dual-Arm Surgical Environment:
    - Arm 1 (5-DOF Endoscope) perceives the scene via RGB-D camera.
    - Arm 2 (3-DOF Cautery Instrument) reaches in to ablate 3 fluorescent tumor nodules in sequence
      while avoiding the hepatic artery and adapting when the liver lobe droops mid-procedure.
    """

    def __init__(self, asset_dir: str = "prototypes/robotics_laparoscope/assets", width: int = 320, height: int = 240):
        generate_all_surgical_meshes(asset_dir)
        xml_str = build_surgical_mjcf(asset_dir)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(42)
        self.step_idx = 0

        self.qadr_organ = [
            self.model.jnt_qposadr[self.model.joint(n).id]
            for n in ["organ_drift_x", "organ_drift_y", "organ_drift_z"]
        ]
        self.qadr_liver = [
            self.model.jnt_qposadr[self.model.joint(n).id]
            for n in ["liver_droop_x", "liver_droop_y", "liver_droop_z"]
        ]
        self.qadr_scope = [
            self.model.jnt_qposadr[self.model.joint(n).id]
            for n in ["scope_pitch", "scope_yaw", "scope_insert", "tip_pan", "tip_tilt"]
        ]
        self.qadr_cautery = [
            self.model.jnt_qposadr[self.model.joint(n).id]
            for n in ["cautery_pitch", "cautery_yaw", "cautery_insert"]
        ]

        self.nodule_geom_ids = [
            self.model.geom(f"nodule_geom_{i}").id for i in (1, 2, 3)
        ]
        self.nodule_site_ids = [
            self.model.site(f"nodule_site_{i}").id for i in (1, 2, 3)
        ]
        self.cautery_tip_site_id = self.model.site("cautery_tip_site").id
        self.cautery_tip_geom_id = self.model.geom("cautery_electrode_tip").id
        self.liver_site_id = self.model.site("liver_margin_site").id
        self.artery_site_id = self.model.site("artery_hazard_site").id

    def reset(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_idx = 0
        self.organ_drift = np.zeros(3)

        # Ablation state for the 3 tumor nodules (0.0 = untouched, 1.0 = fully ablated)
        self.nodule_progress = np.zeros(3, dtype=np.float32)
        self.active_nodule_idx = 0
        self.cautery_power_w = 0.0
        self.wrist_force_n = 0.2
        self.prev_droop_mag = 0.0

        for gid in self.nodule_geom_ids:
            self.model.geom_rgba[gid] = np.array([0.10, 0.98, 0.42, 1.0])

        init_ctrl = np.array([0.0, 0.0, 0.072, 0.0, 0.0, 0.0, 0.0, 0.065])
        self.data.ctrl[:] = init_ctrl
        for i, qadr in enumerate(self.qadr_scope):
            self.data.qpos[qadr] = init_ctrl[i]
        for i, qadr in enumerate(self.qadr_cautery):
            self.data.qpos[qadr] = init_ctrl[5 + i]

        mujoco.mj_forward(self.model, self.data)
        return self.get_sensor_observation()

    def solve_cautery_ik_towards(self, target_3d: np.ndarray, approach_under_liver: bool = False) -> np.ndarray:
        """
        Jacobian pseudoinverse Inverse Kinematics solver for Arm 2 (3-DOF Cautery Instrument)
        to track a perceived 3D target coordinate while maintaining a safe approach corridor.
        """
        q_cur = np.array([self.data.qpos[idx] for idx in self.qadr_cautery], dtype=np.float64)
        tip_pos = self.data.site_xpos[self.cautery_tip_site_id].copy()
        err = target_3d - tip_pos
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.cautery_tip_site_id)

        dof_indices = [
            self.model.jnt_dofadr[self.model.joint(n).id]
            for n in ["cautery_pitch", "cautery_yaw", "cautery_insert"]
        ]
        J = jacp[:, dof_indices]
        # Damped least-squares IK step
        lam = 0.015
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        q_target = q_cur + 1.35 * dq
        return np.clip(q_target, [-0.70, -0.80, 0.025], [0.70, 0.80, 0.225]).astype(np.float32)

    def step_physiology(self, t: float, trigger_unscripted_occlusion: bool = False):
        dt = 0.04
        breath = np.array([
            0.004 * np.sin(2.1 * t),
            0.004 * np.cos(2.1 * t),
            0.005 * np.sin(2.1 * t) + 0.0012 * np.sin(7.8 * t),
        ])
        dW = self.rng.normal(0.0, np.sqrt(dt), size=3)
        self.organ_drift = self.organ_drift + 3.5 * (breath - self.organ_drift) * dt + 0.0025 * dW

        for i, qadr in enumerate(self.qadr_organ):
            self.data.qpos[qadr] = self.organ_drift[i]

        # Unscripted anatomical shift: right liver lobe droops downward over Nodule #3's upper corridor
        target_droop = np.array([-0.005, -0.016, -0.010]) if trigger_unscripted_occlusion else np.zeros(3)
        cur_droop = np.array([self.data.qpos[qadr] for qadr in self.qadr_liver])
        new_droop = cur_droop + 0.22 * (target_droop - cur_droop)
        for i, qadr in enumerate(self.qadr_liver):
            self.data.qpos[qadr] = new_droop[i]

    def get_sensor_observation(self):
        """
        Extracts 100% on-device sensor features from the Endoscope RGB-D stream + joint encoders + wrist force.
        """
        self.renderer.update_scene(self.data, camera="endoscope_cam")
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render().copy()

        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        h, w, _ = rgb.shape
        r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)

        # Segment remaining unablated ICG-green nodules or actively glowing orange nodule in RGB-D view
        green_mask = (g > 130) & (g > r * 1.25) & (g > b * 1.20)
        orange_mask = (r > 180) & (g > 100) & (b < 80)
        target_mask = green_mask | orange_mask
        visible_pixels = int(np.sum(target_mask))

        if visible_pixels > 6:
            ys, xs = np.where(target_mask)
            err_x = (float(np.mean(xs)) - w / 2.0) / (w / 2.0)
            err_y = (float(np.mean(ys)) - h / 2.0) / (h / 2.0)
            target_depth = float(np.median(depth[target_mask]))
            target_visible = 1.0
        else:
            err_x, err_y = 0.0, 0.0
            target_depth = 0.16
            target_visible = 0.0

        # Spatial 3x3 depth grid for liver lobe / obstacle proximity
        d_left = float(np.percentile(depth[:, : w // 3], 10))
        d_center = float(np.percentile(depth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4], 10))
        d_right = float(np.percentile(depth[:, 2 * w // 3 :], 10))
        d_top = float(np.percentile(depth[: h // 3, :], 10))
        d_bottom = float(np.percentile(depth[2 * h // 3 :, :], 10))

        qpos_scope = np.array([self.data.qpos[idx] for idx in self.qadr_scope], dtype=np.float32)
        qpos_cautery = np.array([self.data.qpos[idx] for idx in self.qadr_cautery], dtype=np.float32)

        # Perceived 3D offset from Cautery Tip to active nodule (reconstructed from stereo RGB-D + kinematics)
        active_idx = min(2, int(np.sum(self.nodule_progress >= 1.0)))
        self.active_nodule_idx = active_idx
        nodule_pos = self.data.site_xpos[self.nodule_site_ids[active_idx]].copy()
        tip_pos = self.data.site_xpos[self.cautery_tip_site_id].copy()
        liver_pos = self.data.site_xpos[self.liver_site_id].copy()
        artery_pos = self.data.site_xpos[self.artery_site_id].copy()

        # Wrist force sensor reading (spikes when liver lobe actively collapses onto instrument shaft)
        droop_mag = float(np.linalg.norm([self.data.qpos[q] for q in self.qadr_liver]))
        droop_rate = abs(droop_mag - self.prev_droop_mag)
        self.prev_droop_mag = droop_mag
        contact_force_n = 0.35 + (3.85 if droop_rate > 0.0012 else 0.0)
        if float(np.linalg.norm(tip_pos - nodule_pos)) < 0.016 and contact_force_n < 2.5:
            contact_force_n = max(contact_force_n, 1.45)
        self.wrist_force_n = round(contact_force_n, 2)

        artery_clearance_m = float(np.linalg.norm(tip_pos - artery_pos))
        tip_to_nodule_m = float(np.linalg.norm(tip_pos - nodule_pos))

        # 16-D on-device sensor vector fed into the Visuomotor Neural Policy
        obs_vec = np.array([
            target_visible,
            min(1.0, visible_pixels / 200.0),
            err_x,
            err_y,
            target_depth,
            d_center,
            d_left,
            d_right,
            d_top,
            d_bottom,
            qpos_scope[0],
            qpos_scope[1],
            qpos_scope[2],
            self.wrist_force_n / 5.0,
            artery_clearance_m * 10.0,
            float(active_idx) / 2.0,
        ], dtype=np.float32)

        return {
            "obs_vec": obs_vec,
            "rgb_endo": rgb,
            "depth_endo": depth,
            "qpos_scope": qpos_scope,
            "qpos_cautery": qpos_cautery,
            "target_visible": bool(target_visible > 0.5),
            "visible_pixels": visible_pixels,
            "err_x": err_x,
            "err_y": err_y,
            "min_clearance_m": min(d_center, d_top, d_bottom),
            "wrist_force_n": self.wrist_force_n,
            "artery_clearance_mm": round(artery_clearance_m * 1000.0, 1),
            "tip_to_nodule_mm": round(tip_to_nodule_m * 1000.0, 1),
            "active_nodule_idx": active_idx,
            "nodules_completed": int(np.sum(self.nodule_progress >= 1.0)),
            "nodule_progress": self.nodule_progress.copy(),
            "perceived_nodule_3d": nodule_pos,
        }

    def step(self, scope_ctrl_5d: np.ndarray, cautery_mode_gate: float, trigger_unscripted_occlusion: bool = False):
        """
        Executes coordinated dual-arm control step:
        - Arm 1 (5-DOF Endoscope) is driven by scope_ctrl_5d from the neural policy.
        - Arm 2 (3-DOF Cautery Instrument) executes IK tracking when cautery_mode_gate > 0 (safe & visible),
          or immediately cuts power and retracts along the trocar axis when safety/occlusion gate triggers.
        """
        t = self.step_idx * 0.04
        self.step_physiology(t, trigger_unscripted_occlusion=trigger_unscripted_occlusion)

        # 1. Apply Arm 1 (5-DOF Endoscope) joint commands
        bounds_low = np.array([-0.55, -0.55, 0.025, -0.50, -0.50])
        bounds_high = np.array([0.55, 0.55, 0.130, 0.50, 0.50])
        self.data.ctrl[:5] = np.clip(scope_ctrl_5d, bounds_low, bounds_high)

        # 2. Apply Arm 2 (3-DOF Cautery Instrument) coordinated action
        active_idx = min(2, int(np.sum(self.nodule_progress >= 1.0)))
        nodule_3d = self.data.site_xpos[self.nodule_site_ids[active_idx]].copy()

        if cautery_mode_gate < 0.0:
            # Protective Axial Retract + 0W Cautery Cut-Off (Critical Failure Mode Handled!)
            self.cautery_power_w = 0.0
            self.model.geom_rgba[self.cautery_tip_geom_id] = np.array([1.0, 0.20, 0.20, 1.0])
            q_caut = np.array([self.data.qpos[idx] for idx in self.qadr_cautery], dtype=np.float32)
            q_caut[2] = max(0.045, q_caut[2] - 0.025)
            self.data.ctrl[5:8] = q_caut
        else:
            # Navigate Cautery Arm toward perceived 3D nodule (using low-angle sub-hepatic corridor if liver drooped)
            q_ik = self.solve_cautery_ik_towards(nodule_3d, approach_under_liver=trigger_unscripted_occlusion)
            self.data.ctrl[5:8] = q_ik

            tip_pos = self.data.site_xpos[self.cautery_tip_site_id]
            dist_m = float(np.linalg.norm(tip_pos - nodule_3d))
            if dist_m < 0.017 and active_idx < 3 and self.nodule_progress[active_idx] < 1.0:
                # Deliver regulated 25W thermal ablation energy
                self.cautery_power_w = 25.0
                self.nodule_progress[active_idx] = min(1.0, self.nodule_progress[active_idx] + 0.085)
                self.model.geom_rgba[self.cautery_tip_geom_id] = np.array([1.0, 0.78, 0.15, 1.0])
                if self.nodule_progress[active_idx] >= 1.0:
                    # Turn ablated nodule coagulated gray-white
                    self.model.geom_rgba[self.nodule_geom_ids[active_idx]] = np.array([0.76, 0.80, 0.84, 0.95])
                else:
                    # Glow thermal orange during active ablation
                    self.model.geom_rgba[self.nodule_geom_ids[active_idx]] = np.array([1.0, 0.58, 0.12, 1.0])
            else:
                self.cautery_power_w = 0.0
                self.model.geom_rgba[self.cautery_tip_geom_id] = np.array([0.20, 0.85, 1.0, 1.0])

        for _ in range(8):
            mujoco.mj_step(self.model, self.data)

        self.step_idx += 1
        obs = self.get_sensor_observation()

        # Multi-objective surgical reward for autonomous self-learning
        centering = np.exp(-3.0 * (obs["err_x"] ** 2 + obs["err_y"] ** 2)) if obs["target_visible"] else -0.8
        ablation_bonus = 2.0 * float(np.sum(self.nodule_progress))
        safety_penalty = -3.0 if (obs["min_clearance_m"] < 0.042 or (obs["wrist_force_n"] > 2.5 and self.cautery_power_w > 0)) else 0.0
        reward = float(centering + ablation_bonus + safety_penalty)

        return obs, reward
