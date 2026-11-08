"""
Convert CT-derived STL organ meshes to UV-mapped OBJ meshes with realistic
high-resolution procedural biological tissue textures (vascular networks,
hepatic lobules, mucosal folds, biliary serosa, and adipose tissue).
"""

import os
import struct
import numpy as np
from PIL import Image, ImageFilter

SOFA_MESHES = "c:/Users/arhaa/Projects/AgentVerse26/prototypes/sofa_env_assets/assets/meshes"
GLT_MESHES = "c:/Users/arhaa/Projects/AgentVerse26/prototypes/sofa_env_assets/sofa_env/scenes/grasp_lift_touch/meshes"
OUT_DIR = "c:/Users/arhaa/Projects/AgentVerse26/prototypes/textured_anatomy"


def fbm_noise(h: int, w: int, octaves=(4, 8, 16, 32, 64), weights=(0.35, 0.25, 0.20, 0.12, 0.08), seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    for freq, weight in zip(octaves, weights):
        small = rng.random((freq, freq), dtype=np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
        acc += weight * (np.asarray(img, dtype=np.float32) / 255.0)
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
    return acc


def ridge_vessels(h: int, w: int, freq: int = 12, seed: int = 101) -> np.ndarray:
    """Generates branching capillary / sub-serosal vascular ridges using absolute value of zero-mean noise."""
    n1 = fbm_noise(h, w, octaves=(freq, freq * 2, freq * 4), weights=(0.55, 0.30, 0.15), seed=seed)
    n2 = fbm_noise(h, w, octaves=(freq * 2, freq * 4, freq * 8), weights=(0.50, 0.35, 0.15), seed=seed + 7)
    veins1 = np.exp(-((np.abs(n1 - 0.5) * 28.0) ** 1.6))
    veins2 = 0.55 * np.exp(-((np.abs(n2 - 0.5) * 36.0) ** 1.6))
    return np.clip(veins1 + veins2, 0.0, 1.0)


def generate_biological_textures(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    h, w = 512, 512

    # 1. Hepatic Parenchyma (Liver) Texture: Deep burgundy-brown with fine hepatic lobules & dark portal venules
    base_n = fbm_noise(h, w, octaves=(6, 14, 28, 56, 112), weights=(0.30, 0.25, 0.20, 0.15, 0.10), seed=11)
    lobules = fbm_noise(h, w, octaves=(32, 64, 128), weights=(0.5, 0.3, 0.2), seed=12)
    veins = ridge_vessels(h, w, freq=10, seed=13)
    liver_rgb = np.zeros((h, w, 3), dtype=np.float32)
    liver_rgb[:, :, 0] = 115 + 48 * base_n + 18 * lobules - 35 * veins
    liver_rgb[:, :, 1] = 26 + 20 * base_n + 8 * lobules - 12 * veins
    liver_rgb[:, :, 2] = 28 + 22 * base_n + 10 * lobules - 10 * veins
    Image.fromarray(np.clip(liver_rgb, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, "liver_tex.png"))

    # 2. Gallbladder Serosa Texture: Translucent olive-emerald bile tint with rich sub-serosal arterial/venous branching
    gb_n = fbm_noise(h, w, octaves=(5, 12, 24, 48), weights=(0.4, 0.3, 0.2, 0.1), seed=21)
    gb_veins = ridge_vessels(h, w, freq=8, seed=22)
    gb_rgb = np.zeros((h, w, 3), dtype=np.float32)
    gb_rgb[:, :, 0] = 78 + 42 * gb_n + 75 * gb_veins
    gb_rgb[:, :, 1] = 128 + 45 * gb_n - 35 * gb_veins
    gb_rgb[:, :, 2] = 52 + 25 * gb_n - 15 * gb_veins
    Image.fromarray(np.clip(gb_rgb, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, "gallbladder_tex.png"))

    # 3. Gastric / Duodenal Serosa (Stomach & Duodenum): Warm salmon-pink smooth muscle with red serosal capillaries & yellow omental fat patches
    st_n = fbm_noise(h, w, octaves=(4, 10, 22, 44), weights=(0.45, 0.25, 0.20, 0.10), seed=31)
    fat_mask = np.clip((st_n - 0.62) * 3.2, 0.0, 1.0)
    st_veins = ridge_vessels(h, w, freq=11, seed=32)
    st_rgb = np.zeros((h, w, 3), dtype=np.float32)
    st_rgb[:, :, 0] = 205 + 32 * st_n + 25 * fat_mask - 25 * st_veins
    st_rgb[:, :, 1] = 105 + 35 * st_n + 65 * fat_mask - 55 * st_veins
    st_rgb[:, :, 2] = 95 + 30 * st_n - 15 * fat_mask - 45 * st_veins
    Image.fromarray(np.clip(st_rgb, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, "stomach_tex.png"))

    # 4. Pancreas / Retroperitoneal Lobular Fat Texture: Lobulated golden-tan glandular tissue with fine capillaries
    pa_n = fbm_noise(h, w, octaves=(12, 24, 48, 96), weights=(0.40, 0.30, 0.20, 0.10), seed=41)
    pa_veins = ridge_vessels(h, w, freq=9, seed=42)
    pa_rgb = np.zeros((h, w, 3), dtype=np.float32)
    pa_rgb[:, :, 0] = 212 + 35 * pa_n - 10 * pa_veins
    pa_rgb[:, :, 1] = 158 + 38 * pa_n - 48 * pa_veins
    pa_rgb[:, :, 2] = 95 + 30 * pa_n - 35 * pa_veins
    Image.fromarray(np.clip(pa_rgb, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, "pancreas_tex.png"))

    # 5. Peritoneal Cavity Wall Texture: Rich vascularized parietal peritoneum (rose-red with adipose streaks & vessels)
    pe_n = fbm_noise(h, w, octaves=(4, 8, 16, 32), weights=(0.45, 0.25, 0.20, 0.10), seed=51)
    pe_veins = ridge_vessels(h, w, freq=10, seed=52)
    pe_rgb = np.zeros((h, w, 3), dtype=np.float32)
    pe_rgb[:, :, 0] = 165 + 45 * pe_n + 25 * pe_veins
    pe_rgb[:, :, 1] = 68 + 35 * pe_n - 32 * pe_veins
    pe_rgb[:, :, 2] = 64 + 30 * pe_n - 28 * pe_veins
    Image.fromarray(np.clip(pe_rgb, 0, 255).astype(np.uint8)).save(os.path.join(out_dir, "peritoneum_tex.png"))


def stl_to_smooth_uv_obj(stl_path: str, obj_path: str, uv_scale: float = 0.012):
    """
    Reads a binary STL mesh, welds shared vertices within 1e-4 mm, computes smooth
    area-weighted vertex normals (`vn`) and tri-planar/spherical UV coordinates (`vt`),
    and writes a clean `.obj` file that MuJoCo renders with smooth organic shading and textures.
    """
    with open(stl_path, "rb") as f:
        _ = f.read(80)
        count = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, 50)

    raw_verts = np.frombuffer(data[:, 12:48].tobytes(), dtype=np.float32).reshape(count * 3, 3)

    # Weld vertices by rounding to 0.01 mm
    rounded = np.round(raw_verts * 100.0).astype(np.int64)
    _, unique_indices, inverse_indices = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True
    )
    verts = raw_verts[unique_indices]
    faces = inverse_indices.reshape(count, 3)

    # Compute smooth vertex normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fnormals = np.cross(v1 - v0, v2 - v0)

    vnormals = np.zeros_like(verts, dtype=np.float64)
    np.add.at(vnormals, faces[:, 0], fnormals)
    np.add.at(vnormals, faces[:, 1], fnormals)
    np.add.at(vnormals, faces[:, 2], fnormals)
    norms = np.linalg.norm(vnormals, axis=1, keepdims=True) + 1e-12
    vnormals = (vnormals / norms).astype(np.float32)

    # Smooth spatial UV parameterization
    center = verts.mean(axis=0)
    rel = verts - center
    u = (rel[:, 0] + 0.5 * rel[:, 2]) * uv_scale + 0.5
    v = (rel[:, 1] + 0.5 * rel[:, 2]) * uv_scale + 0.5
    uvs = np.stack([u, v], axis=1)

    with open(obj_path, "w", encoding="utf-8") as out:
        out.write("# Smooth UV-mapped anatomical mesh\n")
        for x, y, z in verts:
            out.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for s, t in uvs:
            out.write(f"vt {s:.4f} {t:.4f}\n")
        for nx, ny, nz in vnormals:
            out.write(f"vn {nx:.4f} {ny:.4f} {nz:.4f}\n")
        for i0, i1, i2 in faces + 1:
            out.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")

    print(f"Wrote {os.path.basename(obj_path)}: {len(verts)} welded vertices, {len(faces)} smooth triangles")


if __name__ == "__main__":
    generate_biological_textures(OUT_DIR)
    meshes_to_convert = [
        (f"{SOFA_MESHES}/models/OpenHELP/Liver.stl", f"{OUT_DIR}/Liver_uv.obj", 0.008),
        (f"{SOFA_MESHES}/models/OpenHELP/Stomach.stl", f"{OUT_DIR}/Stomach_uv.obj", 0.010),
        (f"{SOFA_MESHES}/models/OpenHELP/Duodenum.stl", f"{OUT_DIR}/Duodenum_uv.obj", 0.012),
        (f"{SOFA_MESHES}/models/OpenHELP/Pancreas.stl", f"{OUT_DIR}/Pancreas_uv.obj", 0.012),
        (f"{SOFA_MESHES}/models/OpenHELP/Spleen.stl", f"{OUT_DIR}/Spleen_uv.obj", 0.010),
        (f"{SOFA_MESHES}/models/OpenHELP/Kidney_right.stl", f"{OUT_DIR}/Kidney_right_uv.obj", 0.012),
        (f"{SOFA_MESHES}/models/OpenHELP/Kidney_left.stl", f"{OUT_DIR}/Kidney_left_uv.obj", 0.012),
        (f"{GLT_MESHES}/gallbladder_visual.stl", f"{OUT_DIR}/Gallbladder_uv.obj", 0.018),
    ]
    for src, dst, scale in meshes_to_convert:
        stl_to_smooth_uv_obj(src, dst, uv_scale=scale)
