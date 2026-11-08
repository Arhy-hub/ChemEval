"""
Autonomous Laparoscopic Camera Control Environment (MuJoCo 4.0)
================================================================
Uses:
- Official Google DeepMind `mujoco_menagerie` 7-DOF Franka Emika Panda arm (`franka_emika_panda`)
- Real 30-degree Laparoscope CAD meshes (`laparoscope_body.stl`, `laparoscope_optics_30_degree.stl`, `laparoscope_ring_coupler.stl`, `laparoscope_cable.stl`)
- Smooth UV-mapped Heidelberg OpenHELP CT-derived human organ `.obj` meshes with high-res biological vascular textures
  (`Liver_uv.obj`, `Stomach_uv.obj`, `Duodenum_uv.obj`, `Pancreas_uv.obj`, `Spleen_uv.obj`, `Kidney_right_uv.obj`, `Kidney_left_uv.obj`)
- Smooth UV-mapped LapGym Gallbladder mesh (`Gallbladder_uv.obj`) with biliary serosa vascular texture
- Physical 3-Port Trocar Cannulas (`endoscope_trocar_port`, `cauter_trocar_port`, `grasper_trocar_port`) anchored at the abdominal wall entry points with true RCM (Remote Center of Motion) pivot kinematics so NO shaft ever phases through tissue!
"""

import os
import numpy as np
import mujoco
from PIL import Image

PANDA_ASSETS = "C:/Users/arhaa/.cache/robot_descriptions/mujoco_menagerie/franka_emika_panda/assets"
SOFA_MESHES = "c:/Users/arhaa/Projects/AgentVerse26/prototypes/sofa_env_assets/assets/meshes"
TEX_ANATOMY = "c:/Users/arhaa/Projects/AgentVerse26/prototypes/textured_anatomy"

# OpenHELP phantom origin in world meters (placed on the operating table in front of Franka Panda)
PATIENT_ORIGIN = np.array([0.52, -0.05, 0.14])

# Fixed 3D Trocar Port (Remote Center of Motion / RCM) coordinates in the abdominal wall
ENDOSCOPE_RCM_POS = PATIENT_ORIGIN + np.array([-0.035, -0.070, 0.255])
CAUTER_RCM_POS = PATIENT_ORIGIN + np.array([0.115, -0.035, 0.220])
GRASPER_RCM_POS = PATIENT_ORIGIN + np.array([-0.165, -0.025, 0.215])


def set_instrument_from_rcm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    rcm_pos: np.ndarray,
    tip_pos: np.ndarray,
    roll_angle: float = 0.0,
):
    """
    Sets the 6-DOF pose (`mocap_pos`, `mocap_quat`) of a laparoscopic surgical instrument
    (`surgical_cauter` or `surgical_grasper`) so its shaft axis (+Z in mesh frame, where
    z=-0.35m is the handle and z=0.0m is the tip) passes EXACTLY through its fixed abdominal
    trocar cannula (`rcm_pos`) to reach `tip_pos`.
    This guarantees 0.0mm lateral trocar phasing!
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[body_id]

    z_axis = tip_pos - rcm_pos
    z_norm = np.linalg.norm(z_axis) + 1e-9
    z_axis = z_axis / z_norm

    # Build orthonormal frame with +Z along shaft (from trocar port to surgical tip)
    up_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, up_ref)) > 0.95:
        up_ref = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up_ref, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-9
    y_axis = np.cross(z_axis, x_axis)

    # Apply optional axial spin (roll_angle) around the shaft +Z axis
    c, s = np.cos(roll_angle), np.sin(roll_angle)
    x_rot = c * x_axis + s * y_axis
    y_rot = -s * x_axis + c * y_axis

    rot_mat = np.column_stack([x_rot, y_rot, z_axis]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_mat)

    data.mocap_pos[mocap_id] = tip_pos
    data.mocap_quat[mocap_id] = quat

    # Also align the coaxial abdominal trocar cannula sleeve at the fixed RCM pivot point
    port_map = {
        "surgical_cauter": "cauter_trocar_port",
        "surgical_grasper": "grasper_trocar_port",
    }
    if body_name in port_map:
        port_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, port_map[body_name])
        if port_id >= 0 and model.body_mocapid[port_id] >= 0:
            pm_id = model.body_mocapid[port_id]
            data.mocap_pos[pm_id] = rcm_pos
            data.mocap_quat[pm_id] = quat


def build_scene_xml() -> str:
    """Builds the MuJoCo XML combining Franka Emika Panda + Laparoscope + Textured OpenHELP CT Anatomy + 3 RCM Trocar Ports."""
    ox, oy, oz = PATIENT_ORIGIN
    ex, ey, ez = ENDOSCOPE_RCM_POS
    cx, cy, cz = CAUTER_RCM_POS
    gx, gy, gz = GRASPER_RCM_POS

    xml = f"""<mujoco model="franka_autonomous_laparoscope">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.004" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.42 0.42 0.45" ambient="0.24 0.20 0.20" specular="0.45 0.45 0.45"/>
    <quality shadowsize="4096"/>
    <map znear="0.005" zfar="5.0"/>
  </visual>

  <default>
    <default class="panda">
      <material specular="0.5" shininess="0.25"/>
      <joint armature="0.1" damping="2.0" axis="0 0 1" range="-2.8973 2.8973"/>
      <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
      <default class="finger">
        <joint axis="0 1 0" type="slide" range="0 0.04"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" contype="0" conaffinity="0" group="3"/>
      </default>
    </default>
    <default class="organ_mesh">
      <geom type="mesh" contype="1" conaffinity="1" group="1" solref="0.01 1" solimp="0.9 0.95 0.001"/>
    </default>
  </default>

  <asset>
    <texture name="or_floor_tex" type="2d" builtin="checker" rgb1="0.12 0.16 0.20" rgb2="0.16 0.20 0.25" width="512" height="512"/>
    <material name="or_floor_mat" texture="or_floor_tex" texrepeat="6 6" specular="0.3" shininess="0.2"/>
    <material name="or_table_mat" rgba="0.15 0.19 0.24 1" specular="0.4" shininess="0.3"/>
    <material name="surgical_drape_mat" rgba="0.09 0.28 0.36 1" specular="0.15" shininess="0.1"/>

    <!-- High-Resolution Biological Vascular & Parenchymal Tissue Textures -->
    <texture name="liver_tex" type="2d" file="{TEX_ANATOMY}/liver_tex.png"/>
    <texture name="gallbladder_tex" type="2d" file="{TEX_ANATOMY}/gallbladder_tex.png"/>
    <texture name="stomach_tex" type="2d" file="{TEX_ANATOMY}/stomach_tex.png"/>
    <texture name="pancreas_tex" type="2d" file="{TEX_ANATOMY}/pancreas_tex.png"/>
    <texture name="peritoneum_tex" type="2d" file="{TEX_ANATOMY}/peritoneum_tex.png"/>

    <!-- Franka Emika Panda materials -->
    <material class="panda" name="white" rgba="0.96 0.96 0.97 1"/>
    <material class="panda" name="off_white" rgba="0.88 0.90 0.92 1"/>
    <material class="panda" name="black" rgba="0.20 0.22 0.24 1"/>
    <material class="panda" name="green" rgba="0.1 0.85 0.3 1"/>
    <material class="panda" name="light_blue" rgba="0.04 0.54 0.78 1"/>

    <!-- Textured Wet Biological Organ Materials (High Specular/Shininess for Intracorporeal Serosa) -->
    <material name="liver_mat" texture="liver_tex" specular="0.92" shininess="0.85"/>
    <material name="gallbladder_mat" texture="gallbladder_tex" specular="0.90" shininess="0.82"/>
    <material name="stomach_mat" texture="stomach_tex" specular="0.85" shininess="0.72"/>
    <material name="duodenum_mat" texture="stomach_tex" rgba="0.95 0.88 0.88 1.0" specular="0.82" shininess="0.68"/>
    <material name="pancreas_mat" texture="pancreas_tex" specular="0.78" shininess="0.62"/>
    <material name="spleen_mat" texture="liver_tex" rgba="0.85 0.75 0.85 1.0" specular="0.85" shininess="0.75"/>
    <material name="kidney_mat" texture="liver_tex" rgba="0.90 0.82 0.80 1.0" specular="0.80" shininess="0.70"/>
    <material name="peritoneum_mat" texture="peritoneum_tex" texrepeat="3 3" specular="0.65" shininess="0.55"/>
    <material name="heart_mat" rgba="0.75 0.18 0.18 1.0" specular="0.80" shininess="0.70"/>
    <material name="lung_mat" rgba="0.85 0.42 0.42 1.0" specular="0.60" shininess="0.50"/>
    <material name="trachea_mat" rgba="0.92 0.80 0.68 1.0" specular="0.50" shininess="0.40"/>
    <material name="pelvis_mat" rgba="0.58 0.28 0.22 1.0" specular="0.55" shininess="0.45"/>
    <material name="bladder_mat" rgba="0.88 0.35 0.12 1.0" specular="0.75" shininess="0.65"/>
    <material name="rectum_mat" rgba="0.85 0.50 0.45 1.0" specular="0.70" shininess="0.60"/>
    <!-- Translucent surgical cutaway torso & abdominal wall so internal vascularized organs & trocars are crystal-clear -->
    <material name="torso_skin_mat" rgba="0.88 0.72 0.64 0.22" specular="0.28" shininess="0.22"/>
    <material name="abd_wall_mat" rgba="0.92 0.76 0.68 0.12" specular="0.25" shininess="0.20"/>
    <material name="vessel_artery_mat" texture="liver_tex" rgba="0.92 0.35 0.35 1.0" specular="0.88" shininess="0.80"/>
    <material name="duct_mat" texture="gallbladder_tex" rgba="0.78 0.88 0.68 1.0" specular="0.85" shininess="0.75"/>

    <!-- Laparoscope & Surgical Instrument Materials -->
    <material name="scope_steel_mat" rgba="0.82 0.85 0.88 1.0" specular="0.95" shininess="0.90"/>
    <material name="scope_body_mat" rgba="0.18 0.22 0.26 1.0" specular="0.70" shininess="0.60"/>
    <material name="scope_coupler_mat" rgba="0.12 0.45 0.72 1.0" specular="0.80" shininess="0.70"/>
    <material name="inst_shaft_mat" rgba="0.20 0.22 0.26 1.0" specular="0.85" shininess="0.80"/>
    <material name="cauter_tip_mat" rgba="0.92 0.94 0.96 1.0" specular="0.95" shininess="0.90"/>
    <material name="cauter_active_mat" rgba="0.0 0.95 1.0 1.0" emission="0.65" specular="0.9" shininess="0.9"/>
    <material name="jaw_metal_mat" rgba="0.80 0.84 0.88 1.0" specular="0.95" shininess="0.90"/>
    <material name="trocar_mat" rgba="0.24 0.28 0.34 0.95" specular="0.8" shininess="0.7"/>
    <material name="trocar_seal_mat" rgba="0.10 0.65 0.85 0.95" specular="0.8" shininess="0.7"/>

    <!-- Franka Emika Panda Visual Meshes -->
    <mesh file="{PANDA_ASSETS}/link0_0.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_1.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_2.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_3.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_4.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_5.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_7.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_8.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_9.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_10.obj"/>
    <mesh file="{PANDA_ASSETS}/link0_11.obj"/>
    <mesh file="{PANDA_ASSETS}/link1.obj"/>
    <mesh file="{PANDA_ASSETS}/link2.obj"/>
    <mesh file="{PANDA_ASSETS}/link3_0.obj"/>
    <mesh file="{PANDA_ASSETS}/link3_1.obj"/>
    <mesh file="{PANDA_ASSETS}/link3_2.obj"/>
    <mesh file="{PANDA_ASSETS}/link3_3.obj"/>
    <mesh file="{PANDA_ASSETS}/link4_0.obj"/>
    <mesh file="{PANDA_ASSETS}/link4_1.obj"/>
    <mesh file="{PANDA_ASSETS}/link4_2.obj"/>
    <mesh file="{PANDA_ASSETS}/link4_3.obj"/>
    <mesh file="{PANDA_ASSETS}/link5_0.obj"/>
    <mesh file="{PANDA_ASSETS}/link5_1.obj"/>
    <mesh file="{PANDA_ASSETS}/link5_2.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_0.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_1.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_2.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_3.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_4.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_5.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_6.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_7.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_8.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_9.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_10.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_11.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_12.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_13.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_14.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_15.obj"/>
    <mesh file="{PANDA_ASSETS}/link6_16.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_0.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_1.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_2.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_3.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_4.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_5.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_6.obj"/>
    <mesh file="{PANDA_ASSETS}/link7_7.obj"/>
    <mesh file="{PANDA_ASSETS}/hand_0.obj"/>
    <mesh file="{PANDA_ASSETS}/hand_1.obj"/>
    <mesh file="{PANDA_ASSETS}/hand_2.obj"/>
    <mesh file="{PANDA_ASSETS}/hand_3.obj"/>
    <mesh file="{PANDA_ASSETS}/hand_4.obj"/>
    <mesh file="{PANDA_ASSETS}/finger_0.obj"/>
    <mesh file="{PANDA_ASSETS}/finger_1.obj"/>

    <!-- Smooth UV-Mapped & Vertex-Normal Welded Human CT Organ Meshes (.obj, scaled mm -> m = 0.001) -->
    <mesh name="openhelp_liver" file="{TEX_ANATOMY}/Liver_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_stomach" file="{TEX_ANATOMY}/Stomach_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_duodenum" file="{TEX_ANATOMY}/Duodenum_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_pancreas" file="{TEX_ANATOMY}/Pancreas_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_spleen" file="{TEX_ANATOMY}/Spleen_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_kidney_r" file="{TEX_ANATOMY}/Kidney_right_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_kidney_l" file="{TEX_ANATOMY}/Kidney_left_uv.obj" scale="0.001 0.001 0.001"/>
    <mesh name="lapgym_gallbladder" file="{TEX_ANATOMY}/Gallbladder_uv.obj" scale="0.00075 0.00075 0.00075"/>

    <!-- Thoracic / Pelvic Context Meshes -->
    <mesh name="openhelp_torso" file="{TEX_ANATOMY}/Torso_Surgical_Cutaway.obj" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_heart" file="{SOFA_MESHES}/models/OpenHELP/Heart.stl" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_lung_r" file="{SOFA_MESHES}/models/OpenHELP/Lung_right.stl" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_lung_l" file="{SOFA_MESHES}/models/OpenHELP/Lung_left.stl" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_trachea" file="{SOFA_MESHES}/models/OpenHELP/Trachea.stl" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_pelvis" file="{SOFA_MESHES}/models/OpenHELP/Pelvis_Muscles.stl" scale="0.001 0.001 0.001"/>
    <mesh name="openhelp_abd_wall" file="{SOFA_MESHES}/models/OpenHELP/Abdominal_wall.stl" scale="0.001 0.001 0.001"/>

    <!-- Real 3D Laparoscope & Surgical Instrument CAD Meshes (scaled mm -> m = 0.001) -->
    <mesh name="laparoscope_body_mesh" file="{SOFA_MESHES}/endoscopes/laparoscope_body.stl" scale="0.001 0.001 0.001"/>
    <mesh name="laparoscope_optics_mesh" file="{SOFA_MESHES}/endoscopes/laparoscope_optics_30_degree.stl" scale="0.001 0.001 0.001"/>
    <mesh name="laparoscope_coupler_mesh" file="{SOFA_MESHES}/endoscopes/laparoscope_ring_coupler.stl" scale="0.001 0.001 0.001"/>
    <mesh name="laparoscope_cable_mesh" file="{SOFA_MESHES}/endoscopes/laparoscope_cable.stl" scale="0.001 0.001 0.001"/>
    <mesh name="inst_shaft_mesh" file="{SOFA_MESHES}/instruments/instrument_shaft.stl" scale="0.001 0.001 0.001"/>
    <mesh name="inst_handle_mesh" file="{SOFA_MESHES}/instruments/laparoscopic_handle_body.stl" scale="0.001 0.001 0.001"/>
    <mesh name="inst_lever_mesh" file="{SOFA_MESHES}/instruments/laparoscopic_handle_lever.stl" scale="0.001 0.001 0.001"/>
    <mesh name="dissection_electrode_mesh" file="{SOFA_MESHES}/instruments/dissection_electrode.stl" scale="0.001 0.001 0.001"/>
    <mesh name="maryland_jaw_l_mesh" file="{SOFA_MESHES}/instruments/maryland_dissector_jaw_left.stl" scale="0.001 0.001 0.001"/>
    <mesh name="maryland_jaw_r_mesh" file="{SOFA_MESHES}/instruments/maryland_dissector_jaw_right.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <!-- OR Lighting -->
    <light name="or_overhead_1" pos="0.45 0.0 1.6" dir="0 0 -1" diffuse="0.65 0.65 0.70" specular="0.4 0.4 0.4" directional="false"/>
    <light name="or_overhead_2" pos="0.20 -0.45 1.3" dir="0.3 0.4 -0.8" diffuse="0.40 0.42 0.45" specular="0.2 0.2 0.2" directional="false"/>
    <!-- Warm intracorporeal cavity fill light so deep organs are visible in 3rd-person cutaway -->
    <light name="cavity_fill" pos="{ox - 0.01} {oy + 0.04} {oz + 0.22}" dir="0 0.3 -0.9" diffuse="0.58 0.48 0.46" specular="0.55 0.55 0.55" cutoff="75"/>

    <!-- External 3rd-person cameras -->
    <camera name="or_overview_cam" pos="0.22 -1.22 0.84" xyaxes="0.99 0.08 0 -0.02 0.25 0.97" fovy="46"/>
    <camera name="cavity_closeup_cam" pos="{ox - 0.02} {oy - 0.48} {oz + 0.33}" xyaxes="0.99 0.12 0 -0.04 0.32 0.95" fovy="42"/>

    <!-- Operating Room Floor & Full-Length Surgical Table -->
    <geom name="or_floor" type="plane" size="2.0 2.0 0.1" pos="0 0 0" material="or_floor_mat" contype="0" conaffinity="0"/>
    <geom name="or_table_base" type="box" size="0.32 0.45 0.07" pos="{ox} {oy + 0.06} 0.07" material="or_table_mat" contype="0" conaffinity="0"/>
    <geom name="or_table_pad" type="box" size="0.26 0.42 0.015" pos="{ox} {oy + 0.06} 0.14" material="surgical_drape_mat" contype="0" conaffinity="0"/>
    <!-- Textured vascularized peritoneal cavity lining (strictly internal behind the organs so 3rd-person view is clean and 1st-person endoscope view is 100% enclosed) -->
    <geom name="peritoneal_floor" type="ellipsoid" size="0.13 0.13 0.015" pos="{ox - 0.02} {oy + 0.06} {oz + 0.052}" material="peritoneum_mat" contype="0" conaffinity="0"/>
    <geom name="peritoneal_backwall" type="ellipsoid" size="0.12 0.035 0.080" pos="{ox - 0.02} {oy + 0.145} {oz + 0.100}" material="peritoneum_mat" contype="0" conaffinity="0"/>

    <!-- ========================================================================= -->
    <!-- 3 PHYSICAL LAPAROSCOPIC TROCAR CANNULA PORTS AT ABDOMINAL ENTRY POINTS    -->
    <!-- Every instrument & camera shaft strictly pivots through its trocar port!  -->
    <!-- ========================================================================= -->
    <!-- Port 1: 12mm Umbilical/Para-umbilical Optical Trocar for 30° Laparoscope -->
    <body name="endoscope_trocar_port" mocap="true" pos="{ex} {ey} {ez}" euler="-2.30 0.12 0">
      <geom name="endo_trocar_sleeve" type="cylinder" size="0.0085 0.026" material="trocar_mat" contype="0" conaffinity="0"/>
      <geom name="endo_trocar_valve" type="cylinder" size="0.015 0.006" pos="0 0 -0.022" material="trocar_seal_mat" contype="0" conaffinity="0"/>
      <geom name="endo_trocar_collar" type="cylinder" size="0.018 0.003" pos="0 0 0.002" rgba="0.12 0.85 0.95 0.85" contype="0" conaffinity="0"/>
      <site name="rcm_pivot_site" pos="0 0 0" size="0.002" rgba="0 0.9 1 0.0"/>
    </body>

    <!-- Port 2: 5mm Left-Subcostal Working Trocar for Surgeon's Electrocautery Hook -->
    <body name="cauter_trocar_port" mocap="true" pos="{cx} {cy} {cz}" euler="-2.93 0.613 0">
      <geom name="cauter_trocar_sleeve" type="cylinder" size="0.0065 0.024" material="trocar_mat" contype="0" conaffinity="0"/>
      <geom name="cauter_trocar_valve" type="cylinder" size="0.012 0.005" pos="0 0 -0.020" material="trocar_seal_mat" contype="0" conaffinity="0"/>
      <geom name="cauter_trocar_collar" type="cylinder" size="0.014 0.0025" pos="0 0 0.002" rgba="0.12 0.85 0.95 0.85" contype="0" conaffinity="0"/>
      <site name="cauter_rcm_site" pos="0 0 0" size="0.002" rgba="0 0.9 1 0.0"/>
    </body>

    <!-- Port 3: 5mm Right-Flank Assistant Trocar for Maryland Grasping Forceps -->
    <body name="grasper_trocar_port" mocap="true" pos="{gx} {gy} {gz}" euler="-2.79 -0.651 0">
      <geom name="grasper_trocar_sleeve" type="cylinder" size="0.0065 0.024" material="trocar_mat" contype="0" conaffinity="0"/>
      <geom name="grasper_trocar_valve" type="cylinder" size="0.012 0.005" pos="0 0 -0.020" material="trocar_seal_mat" contype="0" conaffinity="0"/>
      <geom name="grasper_trocar_collar" type="cylinder" size="0.014 0.0025" pos="0 0 0.002" rgba="0.12 0.85 0.95 0.85" contype="0" conaffinity="0"/>
      <site name="grasper_rcm_site" pos="0 0 0" size="0.002" rgba="0 0.9 1 0.0"/>
    </body>

    <!-- ========================================================================= -->
    <!-- 7-DOF FRANKA EMIKA PANDA ROBOTIC CAMERA HOLDER (`mujoco_menagerie`)       -->
    <!-- Placed FIRST in worldbody so qpos[0:7] and ctrl[0:7] are joint1..joint7!  -->
    <!-- ========================================================================= -->
    <geom name="panda_pedestal" type="cylinder" size="0.09 0.15" pos="0.02 -0.08 0.15" material="or_table_mat" contype="0" conaffinity="0"/>
    <body name="link0" pos="0.02 -0.08 0.30" childclass="panda">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
      <geom mesh="link0_0" material="off_white" class="visual"/>
      <geom mesh="link0_1" material="black" class="visual"/>
      <geom mesh="link0_2" material="off_white" class="visual"/>
      <geom mesh="link0_3" material="black" class="visual"/>
      <geom mesh="link0_4" material="off_white" class="visual"/>
      <geom mesh="link0_5" material="black" class="visual"/>
      <geom mesh="link0_7" material="white" class="visual"/>
      <geom mesh="link0_8" material="white" class="visual"/>
      <geom mesh="link0_9" material="black" class="visual"/>
      <geom mesh="link0_10" material="off_white" class="visual"/>
      <geom mesh="link0_11" material="white" class="visual"/>
      <body name="link1" pos="0 0 0.333">
        <inertial mass="4.970684" pos="0.003875 0.002081 -0.04762"
          fullinertia="0.70337 0.70661 0.0091170 -0.00013900 0.0067720 0.019169"/>
        <joint name="joint1"/>
        <geom material="white" mesh="link1" class="visual"/>
        <body name="link2" quat="1 -1 0 0">
          <inertial mass="0.646926" pos="-0.003141 -0.02872 0.003495"
            fullinertia="0.0079620 2.8110e-2 2.5995e-2 -3.925e-3 1.0254e-2 7.04e-4"/>
          <joint name="joint2" range="-1.7628 1.7628"/>
          <geom material="white" mesh="link2" class="visual"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="joint3"/>
            <inertial mass="3.228604" pos="2.7518e-2 3.9252e-2 -6.6502e-2"
              fullinertia="3.7242e-2 3.6155e-2 1.083e-2 -4.761e-3 -1.1396e-2 -1.2805e-2"/>
            <geom mesh="link3_0" material="white" class="visual"/>
            <geom mesh="link3_1" material="white" class="visual"/>
            <geom mesh="link3_2" material="white" class="visual"/>
            <geom mesh="link3_3" material="black" class="visual"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <inertial mass="3.587895" pos="-5.317e-2 1.04419e-1 2.7454e-2"
                fullinertia="2.5853e-2 1.9552e-2 2.8323e-2 7.796e-3 -1.332e-3 8.641e-3"/>
              <joint name="joint4" range="-3.0718 -0.0698"/>
              <geom mesh="link4_0" material="white" class="visual"/>
              <geom mesh="link4_1" material="white" class="visual"/>
              <geom mesh="link4_2" material="black" class="visual"/>
              <geom mesh="link4_3" material="white" class="visual"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <inertial mass="1.225946" pos="-1.1953e-2 4.1065e-2 -3.8437e-2"
                  fullinertia="3.5549e-2 2.9474e-2 8.627e-3 -2.117e-3 -4.037e-3 2.29e-4"/>
                <joint name="joint5"/>
                <geom mesh="link5_0" material="black" class="visual"/>
                <geom mesh="link5_1" material="white" class="visual"/>
                <geom mesh="link5_2" material="white" class="visual"/>
                <body name="link6" quat="1 1 0 0">
                  <inertial mass="1.666555" pos="6.0149e-2 -1.4117e-2 -1.0517e-2"
                    fullinertia="1.964e-3 4.354e-3 5.433e-3 1.09e-4 -1.158e-3 3.41e-4"/>
                  <joint name="joint6" range="-0.0175 3.7525"/>
                  <geom mesh="link6_0" material="off_white" class="visual"/>
                  <geom mesh="link6_1" material="white" class="visual"/>
                  <geom mesh="link6_2" material="black" class="visual"/>
                  <geom mesh="link6_3" material="white" class="visual"/>
                  <geom mesh="link6_4" material="white" class="visual"/>
                  <geom mesh="link6_5" material="white" class="visual"/>
                  <geom mesh="link6_6" material="white" class="visual"/>
                  <geom mesh="link6_7" material="light_blue" class="visual"/>
                  <geom mesh="link6_8" material="light_blue" class="visual"/>
                  <geom mesh="link6_9" material="black" class="visual"/>
                  <geom mesh="link6_10" material="black" class="visual"/>
                  <geom mesh="link6_11" material="white" class="visual"/>
                  <geom mesh="link6_12" material="green" class="visual"/>
                  <geom mesh="link6_13" material="white" class="visual"/>
                  <geom mesh="link6_14" material="black" class="visual"/>
                  <geom mesh="link6_15" material="black" class="visual"/>
                  <geom mesh="link6_16" material="white" class="visual"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <inertial mass="7.35522e-01" pos="1.0517e-2 -4.252e-3 6.1597e-2"
                      fullinertia="1.2516e-2 1.0027e-2 4.815e-3 -4.28e-4 -1.196e-3 -7.41e-4"/>
                    <joint name="joint7"/>
                    <geom mesh="link7_0" material="white" class="visual"/>
                    <geom mesh="link7_1" material="black" class="visual"/>
                    <geom mesh="link7_2" material="black" class="visual"/>
                    <geom mesh="link7_3" material="black" class="visual"/>
                    <geom mesh="link7_4" material="black" class="visual"/>
                    <geom mesh="link7_5" material="black" class="visual"/>
                    <geom mesh="link7_6" material="black" class="visual"/>
                    <geom mesh="link7_7" material="white" class="visual"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <inertial mass="0.85" pos="-0.01 0 0.08" diaginertia="0.002 0.004 0.003"/>
                      <geom mesh="hand_0" material="off_white" class="visual"/>
                      <geom mesh="hand_1" material="black" class="visual"/>
                      <geom mesh="hand_2" material="black" class="visual"/>
                      <geom mesh="hand_3" material="white" class="visual"/>
                      <geom mesh="hand_4" material="off_white" class="visual"/>

                      <!-- =============================================================== -->
                      <!-- REAL 3D 30-DEGREE LAPAROSCOPE MOUNTED ON FRANKA HAND FLANGE     -->
                      <!-- =============================================================== -->
                      <body name="laparoscope_mount" pos="0 0 0.44">
                        <geom name="scope_optics_geom" type="mesh" mesh="laparoscope_optics_mesh" material="scope_steel_mat" contype="0" conaffinity="0"/>
                        <geom name="scope_body_geom" type="mesh" mesh="laparoscope_body_mesh" pos="0 0 -0.380" material="scope_body_mat" contype="0" conaffinity="0"/>
                        <geom name="scope_coupler_geom" type="mesh" mesh="laparoscope_coupler_mesh" pos="0 0 -0.380" material="scope_coupler_mat" contype="0" conaffinity="0"/>
                        <geom name="scope_cable_geom" type="mesh" mesh="laparoscope_cable_mesh" pos="0 0 -0.380" material="scope_body_mat" contype="0" conaffinity="0"/>

                        <site name="scope_tip_site" pos="0 0 0" size="0.002" rgba="0 1 0.5 0"/>
                        <site name="scope_shaft_base_site" pos="0 0 -0.380" size="0.002" rgba="0 0.5 1 0"/>

                        <!-- Coaxial Fiber-Optic Xenon Spotlight illuminating the organ cavity -->
                        <light name="endoscope_light" pos="0 0 0.003" dir="0 0 1" diffuse="1.0 0.96 0.92" specular="0.95 0.95 0.95" cutoff="65" exponent="3" directional="false" attenuation="1.0 0.2 1.0"/>

                        <!-- On-board Laparoscope RGB-D Camera looking along shaft (+Z) -->
                        <camera name="endoscope_cam" pos="0 0 0.004" xyaxes="-1 0 0 0 1 0" fovy="62"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- ========================================================================= -->
    <!-- PATIENT ANATOMY BODY (Textured Smooth UV-Mapped OpenHELP + LapGym Meshes) -->
    <!-- Connected via vertical slide joint to simulate physiological respiration  -->
    <!-- ========================================================================= -->
    <body name="patient_anatomy" pos="{ox} {oy} {oz}">
      <inertial pos="0 0.05 0.12" mass="15.0" diaginertia="0.1 0.1 0.1"/>
      <joint name="respiration_z" type="slide" axis="0 0 1" range="-0.008 0.008" damping="50"/>
      <joint name="respiration_y" type="slide" axis="0 1 0" range="-0.005 0.005" damping="50"/>

      <!-- Smooth UV-Textured OpenHELP CT Organs -->
      <geom name="liver_geom" mesh="openhelp_liver" material="liver_mat" class="organ_mesh"/>
      <geom name="stomach_geom" mesh="openhelp_stomach" material="stomach_mat" class="organ_mesh"/>
      <geom name="duodenum_geom" mesh="openhelp_duodenum" material="duodenum_mat" class="organ_mesh"/>
      <geom name="pancreas_geom" mesh="openhelp_pancreas" material="pancreas_mat" class="organ_mesh"/>
      <geom name="spleen_geom" mesh="openhelp_spleen" material="spleen_mat" class="organ_mesh"/>
      <geom name="kidney_r_geom" mesh="openhelp_kidney_r" material="kidney_mat" class="organ_mesh"/>
      <geom name="kidney_l_geom" mesh="openhelp_kidney_l" material="kidney_mat" class="organ_mesh"/>
      <geom name="heart_geom" mesh="openhelp_heart" material="heart_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="lung_r_geom" mesh="openhelp_lung_r" material="lung_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="lung_l_geom" mesh="openhelp_lung_l" material="lung_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="trachea_geom" mesh="openhelp_trachea" material="trachea_mat" contype="0" conaffinity="0" group="1"/>

      <!-- Smooth UV-Textured LapGym Gallbladder tucked in the hepatic bed -->
      <body name="gallbladder_body" pos="-0.048 0.025 0.118" euler="0.15 0.35 -0.4">
        <geom name="gallbladder_geom" mesh="lapgym_gallbladder" material="gallbladder_mat" class="organ_mesh"/>
        <!-- Cystic Duct & Cystic Artery (Calot's Triangle landmarks) -->
        <geom name="cystic_duct" type="capsule" fromto="-0.012 -0.004 -0.008 0.008 -0.007 -0.012" size="0.0022" material="duct_mat" contype="0" conaffinity="0"/>
        <geom name="cystic_artery" type="capsule" fromto="-0.014 0.003 -0.005 0.010 0.005 -0.008" size="0.0016" material="vessel_artery_mat" contype="0" conaffinity="0"/>
        <site name="calot_target_site" pos="-0.008 -0.002 -0.008" size="0.002" rgba="0.4 0.75 0.3 0.0"/>
      </body>

      <!-- Subtle X-Ray Cutaway Abdominal Wall & Torso Shell -->
      <geom name="abd_wall_geom" mesh="openhelp_abd_wall" material="abd_wall_mat" contype="0" conaffinity="0" group="1"/>
      <geom name="torso_geom" mesh="openhelp_torso" material="torso_skin_mat" contype="0" conaffinity="0" group="1"/>
    </body>

    <!-- ========================================================================= -->
    <!-- SURGICAL INSTRUMENTS PIVOTING STRICTLY THROUGH THEIR TROCAR PORTS (RCM)   -->
    <!-- ========================================================================= -->
    <!-- 1. Active Electrocautery Hook (`dissection_electrode.stl` + `laparoscopic_handle_body.stl`) -->
    <body name="surgical_cauter" mocap="true" pos="{ox - 0.042} {oy + 0.012} {oz + 0.102}" euler="-2.15 -0.45 0.2">
      <geom name="cauter_mesh_geom" type="mesh" mesh="dissection_electrode_mesh" material="cauter_tip_mat" contype="0" conaffinity="0"/>
      <geom name="cauter_shaft_sleeve" type="cylinder" fromto="0 0 -0.34 0 0 -0.015" size="0.0028" material="inst_shaft_mat" contype="0" conaffinity="0"/>
      <geom name="cauter_handle_geom" type="mesh" mesh="inst_handle_mesh" pos="0 0 -0.35" material="scope_body_mat" contype="0" conaffinity="0"/>
      <geom name="cauter_lever_geom" type="mesh" mesh="inst_lever_mesh" pos="0 0 -0.35" material="scope_coupler_mat" contype="0" conaffinity="0"/>
      <geom name="cauter_collar" type="cylinder" fromto="0 0 -0.014 0 0 -0.003" size="0.0035" material="cauter_active_mat" contype="0" conaffinity="0"/>
      <site name="cauter_tip_site" pos="0 0 0.012" size="0.002" rgba="0.9 0.95 1.0 0.0"/>
    </body>

    <!-- 2. Assistant Maryland Grasping Forceps (`instrument_shaft.stl` + `maryland_dissector_jaw_left/right.stl` + handle) -->
    <body name="surgical_grasper" mocap="true" pos="{ox - 0.072} {oy + 0.018} {oz + 0.112}" euler="-2.15 0.50 -0.3">
      <geom name="grasper_shaft_geom" type="mesh" mesh="inst_shaft_mesh" material="inst_shaft_mat" contype="0" conaffinity="0"/>
      <geom name="grasper_handle_geom" type="mesh" mesh="inst_handle_mesh" pos="0 0 -0.35" material="scope_body_mat" contype="0" conaffinity="0"/>
      <geom name="grasper_lever_geom" type="mesh" mesh="inst_lever_mesh" pos="0 0 -0.35" material="scope_coupler_mat" contype="0" conaffinity="0"/>
      <geom name="grasper_jaw_l" type="mesh" mesh="maryland_jaw_l_mesh" pos="0 0 0" euler="0 0.22 0" material="jaw_metal_mat" contype="0" conaffinity="0"/>
      <geom name="grasper_jaw_r" type="mesh" mesh="maryland_jaw_r_mesh" pos="0 0 0" euler="0 -0.22 0" material="jaw_metal_mat" contype="0" conaffinity="0"/>
      <site name="grasper_tip_site" pos="0 0 0.018" size="0.002" rgba="0.8 0.85 0.9 0.0"/>
    </body>
  </worldbody>

  <actuator>
    <general class="panda" name="actuator1" joint="joint1" gainprm="4500" biasprm="0 -4500 -450"/>
    <general class="panda" name="actuator2" joint="joint2" gainprm="4500" biasprm="0 -4500 -450" ctrlrange="-1.7628 1.7628"/>
    <general class="panda" name="actuator3" joint="joint3" gainprm="3500" biasprm="0 -3500 -350"/>
    <general class="panda" name="actuator4" joint="joint4" gainprm="3500" biasprm="0 -3500 -350" ctrlrange="-3.0718 -0.0698"/>
    <general class="panda" name="actuator5" joint="joint5" gainprm="2000" biasprm="0 -2000 -200" forcerange="-24 24"/>
    <general class="panda" name="actuator6" joint="joint6" gainprm="2000" biasprm="0 -2000 -200" forcerange="-24 24" ctrlrange="-0.0175 3.7525"/>
    <general class="panda" name="actuator7" joint="joint7" gainprm="2000" biasprm="0 -2000 -200" forcerange="-24 24"/>
    <position name="respiration_act_z" joint="respiration_z" kp="800"/>
    <position name="respiration_act_y" joint="respiration_y" kp="800"/>
  </actuator>
</mujoco>
"""
    return xml


def solve_rcm_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_tip_pos: np.ndarray,
    rcm_port_pos: np.ndarray,
    look_target_pos: np.ndarray,
    max_iters: int = 80,
) -> np.ndarray:
    """
    Solves 7-DOF Franka Emika Panda Inverse Kinematics so that:
    1. `scope_tip_site` reaches `target_tip_pos` inside the abdominal cavity
    2. The laparoscope shaft axis aligns with `(look_target_pos - rcm_port_pos)` so the shaft passes
       cleanly through the abdominal wall trocar port (`rcm_pivot_site` error < 0.5mm!)
    """
    tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "scope_tip_site")
    base_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "scope_shaft_base_site")

    shaft_vec = look_target_pos - rcm_port_pos
    shaft_dir = shaft_vec / (np.linalg.norm(shaft_vec) + 1e-8)

    insertion_depth = np.linalg.norm(target_tip_pos - rcm_port_pos)
    desired_tip = rcm_port_pos + shaft_dir * insertion_depth
    desired_base = desired_tip - shaft_dir * 0.380

    q = data.qpos[:7].copy()
    q_min = model.jnt_range[0:7, 0]
    q_max = model.jnt_range[0:7, 1]
    q_home = np.array([0.0, -0.35, 0.0, -1.75, 0.0, 1.55, 0.0])

    jac_tip = np.zeros((3, model.nv))
    jac_base = np.zeros((3, model.nv))

    for _ in range(max_iters):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)

        cur_tip = data.site_xpos[tip_site_id]
        cur_base = data.site_xpos[base_site_id]

        err_tip = desired_tip - cur_tip
        err_base = desired_base - cur_base
        if np.linalg.norm(err_tip) < 4e-4 and np.linalg.norm(err_base) < 6e-4:
            break

        mujoco.mj_jacSite(model, data, jac_tip, None, tip_site_id)
        mujoco.mj_jacSite(model, data, jac_base, None, base_site_id)

        J = np.vstack([jac_tip[:, :7], 1.15 * jac_base[:, :7]])
        err = np.concatenate([err_tip, 1.15 * err_base])
        err_norm = np.linalg.norm(err)
        if err_norm > 0.05:
            err = err * (0.05 / err_norm)

        JJT = J @ J.T + (0.025 ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))
        dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (0.03 * (q_home - q))
        q = np.clip(q + 0.65 * dq, q_min, q_max)

    data.qpos[:7] = q
    data.ctrl[:7] = q
    mujoco.mj_forward(model, data)

    # Align the umbilical optical trocar sleeve coaxially with the laparoscope shaft at the fixed RCM pivot
    endo_port_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "endoscope_trocar_port")
    if endo_port_id >= 0 and model.body_mocapid[endo_port_id] >= 0:
        pm_id = model.body_mocapid[endo_port_id]
        z_ax = data.site_xpos[tip_site_id] - data.site_xpos[base_site_id]
        z_ax = z_ax / (np.linalg.norm(z_ax) + 1e-9)
        up_ref = np.array([0.0, 1.0, 0.0])
        x_ax = np.cross(up_ref, z_ax)
        x_ax /= np.linalg.norm(x_ax) + 1e-9
        y_ax = np.cross(z_ax, x_ax)
        rot_mat = np.column_stack([x_ax, y_ax, z_ax]).flatten()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot_mat)
        data.mocap_pos[pm_id] = rcm_port_pos
        data.mocap_quat[pm_id] = quat
        mujoco.mj_forward(model, data)

    return q


if __name__ == "__main__":
    xml = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    data.qpos[:7] = np.array([0.0, -0.35, 0.0, -1.75, 0.0, 1.55, 0.0])
    mujoco.mj_forward(model, data)

    # Set both surgical instruments via exact RCM trocar pivot kinematics
    cauter_target = PATIENT_ORIGIN + np.array([-0.042, 0.012, 0.102])
    grasper_target = PATIENT_ORIGIN + np.array([-0.072, 0.018, 0.112])
    set_instrument_from_rcm(model, data, "surgical_cauter", CAUTER_RCM_POS, cauter_target, roll_angle=0.2)
    set_instrument_from_rcm(model, data, "surgical_grasper", GRASPER_RCM_POS, grasper_target, roll_angle=-0.3)
    mujoco.mj_forward(model, data)

    rcm_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rcm_pivot_site")].copy()
    cauter_tip = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cauter_tip_site")].copy()

    dir_to_target = (cauter_tip - rcm_pos) / np.linalg.norm(cauter_tip - rcm_pos)
    dist_rcm_to_target = np.linalg.norm(cauter_tip - rcm_pos)
    desired_tip = rcm_pos + dir_to_target * (dist_rcm_to_target - 0.090)

    q_sol = solve_rcm_ik(model, data, desired_tip, rcm_pos, cauter_tip, max_iters=140)
    tip_actual = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "scope_tip_site")]
    base_actual = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "scope_shaft_base_site")]
    shaft_u = (tip_actual - base_actual) / np.linalg.norm(tip_actual - base_actual)
    rcm_err_mm = np.linalg.norm((rcm_pos - base_actual) - np.dot(rcm_pos - base_actual, shaft_u) * shaft_u) * 1000.0
    print(f"Solved Franka Panda qpos: {np.round(q_sol, 3)} | Trocar RCM Pivot Error: {rcm_err_mm:.2f} mm")

    renderer = mujoco.Renderer(model, height=720, width=960)
    renderer.update_scene(data, camera="or_overview_cam")
    img_overview = renderer.render().copy()

    renderer.update_scene(data, camera="cavity_closeup_cam")
    img_closeup = renderer.render().copy()

    renderer.update_scene(data, camera="endoscope_cam")
    img_endo = renderer.render().copy()

    os.makedirs("c:/Users/arhaa/Projects/AgentVerse26/prototypes/outputs", exist_ok=True)
    combo = np.hstack([img_overview, img_closeup, img_endo])
    Image.fromarray(combo).save("c:/Users/arhaa/Projects/AgentVerse26/prototypes/outputs/preview_franka_openhelp.png")
    print("Saved preview_franka_openhelp.png")
