"""
Anatomical 3D Mesh Generator for Surgical Laparoscopy Simulation in MuJoCo.
Generates high-resolution binary STL meshes of:
1. Human Liver Right/Left Lobes with Falciform Cleft & Gallbladder Fossa (liver_lobe.stl)
2. Bifurcating Hepatic Arterial Tree (hepatic_artery_tree.stl)
3. Pear-shaped Gallbladder / Target Cystic Lesion Bed (gallbladder_bed.stl)
4. Curved Peritoneal Ribcage Arch (rib_arch.stl)
"""

import os
import struct
import numpy as np


def write_binary_stl(filepath: str, vertices: np.ndarray, faces: np.ndarray):
    """Writes a triangle mesh (V, F) to binary STL format with computed face normals."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms

    with open(filepath, "wb") as f:
        header = b"Anatomical Surgical Mesh - Agentverse 2026".ljust(80, b"\0")
        f.write(header)
        f.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *v0[i]))
            f.write(struct.pack("<3f", *v1[i]))
            f.write(struct.pack("<3f", *v2[i]))
            f.write(struct.pack("<H", 0))


def build_uv_sphere_grid(n_lat: int = 48, n_lon: int = 64):
    """Creates a closed UV sphere vertex/face topology that can be sculpted into anatomical organs."""
    verts = []
    # North pole
    verts.append([0.0, 0.0, 1.0])
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            verts.append([x, y, z])
    # South pole
    verts.append([0.0, 0.0, -1.0])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    # Top cap
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + jn])
    # Middle quads -> 2 triangles each
    for i in range(n_lat - 2):
        row1 = 1 + i * n_lon
        row2 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([row1 + j, row2 + j, row2 + jn])
            faces.append([row1 + j, row2 + jn, row1 + jn])
    # Bottom cap
    south_idx = len(verts) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([row2_idx := last_row + j, south_idx, last_row + jn])

    return verts, np.array(faces, dtype=np.int32)


def generate_anatomical_liver_mesh(filepath: str):
    """
    Sculpts a realistic bilobed human liver mesh with:
    - Larger dome-shaped right lobe and tapered left lateral segment
    - Falciform ligament fissure notch
    - Visceral concavity underneath (gallbladder fossa)
    - Organic surface lobulation micro-relief
    """
    verts, faces = build_uv_sphere_grid(n_lat=52, n_lon=72)
    x, y, z = verts[:, 0].copy(), verts[:, 1].copy(), verts[:, 2].copy()

    # Asymmetric bilobed scaling (right lobe fuller, left lobe tapered wedges)
    taper_left = 1.0 - 0.35 * np.clip(x, 0.0, 1.0)
    rx = 0.068 * (1.0 + 0.15 * np.tanh(-2.0 * x))
    ry = 0.044 * taper_left
    rz = 0.026 * taper_left

    x_s = x * rx
    y_s = y * ry
    z_s = z * rz

    # Falciform ligament fissure cleft near x = 0.01
    cleft = np.exp(-((x_s - 0.008) / 0.009) ** 2) * np.clip(y, -0.2, 1.0)
    y_s -= 0.008 * cleft
    z_s -= 0.005 * cleft

    # Visceral underside concavity (fossa for gallbladder & portal triad)
    underside = np.clip(-z, 0.0, 1.0) * np.exp(-((x_s + 0.012) / 0.035) ** 2)
    z_s += 0.009 * underside

    # Subtle organic hepatic parenchyma surface undulation
    ripple = 0.0012 * (np.sin(38.0 * x_s) * np.cos(42.0 * y_s) + 0.5 * np.sin(65.0 * (x_s + y_s)))
    z_s += ripple

    sculpted = np.stack([x_s, y_s, z_s], axis=1).astype(np.float32)
    write_binary_stl(filepath, sculpted, faces)


def generate_gallbladder_mesh(filepath: str):
    """Generates a pear-shaped gallbladder fundus/infundibulum mesh."""
    verts, faces = build_uv_sphere_grid(n_lat=36, n_lon=48)
    x, y, z = verts[:, 0].copy(), verts[:, 1].copy(), verts[:, 2].copy()

    # Pear profile along X axis (neck tapers toward cystic duct)
    neck_profile = 0.65 + 0.45 * np.tanh(-1.8 * x)
    x_s = x * 0.028
    y_s = y * 0.013 * neck_profile + 0.003 * np.sin(2.5 * x)
    z_s = z * 0.012 * neck_profile

    sculpted = np.stack([x_s, y_s, z_s], axis=1).astype(np.float32)
    write_binary_stl(filepath, sculpted, faces)


def build_tube_mesh(centerline: np.ndarray, radii: np.ndarray, n_circ: int = 24):
    """Sweeps a closed tubular vessel mesh along a 3D centerline with varying radius."""
    n_pts = len(centerline)
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

    verts = []
    # Tip cap start
    verts.append(centerline[0])
    up = np.array([0.0, 0.0, 1.0])
    for i in range(n_pts):
        t = tangents[i]
        n1 = np.cross(t, up)
        if np.linalg.norm(n1) < 1e-6:
            n1 = np.cross(t, np.array([0.0, 1.0, 0.0]))
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(t, n1)
        r = radii[i]
        for j in range(n_circ):
            ang = 2.0 * np.pi * j / n_circ
            pt = centerline[i] + r * (np.cos(ang) * n1 + np.sin(ang) * n2)
            verts.append(pt)
    # Tip cap end
    verts.append(centerline[-1])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for j in range(n_circ):
        jn = (j + 1) % n_circ
        faces.append([0, 1 + jn, 1 + j])
    for i in range(n_pts - 1):
        r1 = 1 + i * n_circ
        r2 = 1 + (i + 1) * n_circ
        for j in range(n_circ):
            jn = (j + 1) % n_circ
            faces.append([r1 + j, r2 + j, r2 + jn])
            faces.append([r1 + j, r2 + jn, r1 + jn])
    end_idx = len(verts) - 1
    last_r = 1 + (n_pts - 1) * n_circ
    for j in range(n_circ):
        jn = (j + 1) % n_circ
        faces.append([last_r + j, end_idx, last_r + jn])

    return verts, np.array(faces, dtype=np.int32)


def generate_hepatic_artery_tree(filepath: str):
    """Generates a tortuous bifurcating hepatic artery + cystic artery branch."""
    s = np.linspace(-1.0, 1.0, 45)
    # Main hepatic artery trunk
    c1 = np.stack([
        0.075 * s,
        0.012 * np.sin(3.5 * s) - 0.008,
        0.006 * np.cos(2.8 * s),
    ], axis=1)
    r1 = 0.0065 * (1.0 - 0.18 * s)
    v1, f1 = build_tube_mesh(c1, r1, n_circ=22)

    # Superior cystic/right hepatic branch bifurcating from s=0
    s2 = np.linspace(0.0, 1.0, 30)
    c2 = np.stack([
        0.055 * s2,
        0.012 * np.sin(0.0) - 0.008 + 0.028 * (s2 ** 1.2),
        0.006 + 0.014 * np.sin(2.2 * s2),
    ], axis=1)
    r2 = 0.0048 * (1.0 - 0.3 * s2)
    v2, f2 = build_tube_mesh(c2, r2, n_circ=20)

    verts = np.vstack([v1, v2])
    faces = np.vstack([f1, f2 + len(v1)])
    write_binary_stl(filepath, verts, faces)


def generate_all_surgical_meshes(mesh_dir: str = "prototypes/robotics_laparoscope/assets"):
    os.makedirs(mesh_dir, exist_ok=True)
    liver_path = os.path.join(mesh_dir, "liver_lobe.stl")
    gb_path = os.path.join(mesh_dir, "gallbladder_bed.stl")
    artery_path = os.path.join(mesh_dir, "hepatic_artery_tree.stl")

    generate_anatomical_liver_mesh(liver_path)
    generate_gallbladder_mesh(gb_path)
    generate_hepatic_artery_tree(artery_path)
    return {
        "liver": liver_path,
        "gallbladder": gb_path,
        "artery": artery_path,
    }


if __name__ == "__main__":
    paths = generate_all_surgical_meshes()
    print("Generated high-res anatomical 3D STL meshes:", paths)
