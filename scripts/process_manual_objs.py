import os
import trimesh
import open3d as o3d
import numpy as np
from pathlib import Path

PHONE_TARGET_LENGTH_M = 0.155  # 155 mm phone reference length

def process_manual_dataset():
    dataset_dir = Path("dataset")
    out_dir = Path("output/components")
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"Found {len(folders)} folders in {dataset_dir.resolve()}...\n")

    processed_count = 0
    for folder in sorted(folders, key=lambda x: int(x.name) if x.name.isdigit() else x.name):
        whole_obj = folder / "model_whole.obj"
        if not whole_obj.exists():
            print(f"[SKIP] {folder.name}: 'model_whole.obj' not found (needed for scale).")
            continue

        # 1. Load the full phone mesh to compute the true global scale factor
        whole_mesh = trimesh.load(str(whole_obj), force="mesh")
        if isinstance(whole_mesh, trimesh.Scene):
            whole_mesh = whole_mesh.dump(concatenate=True)

        scale_factor = PHONE_TARGET_LENGTH_M / (np.max(whole_mesh.extents) + 1e-8)

        # 2. Process all component OBJ meshes
        for obj_path in folder.glob("model_*.obj"):
            comp_name = obj_path.stem.replace("model_", "")

            # Exclude non-target global/chassis meshes and screws
            if comp_name.lower() in ["whole", "other", "screw"]:
                continue

            comp_mesh = trimesh.load(str(obj_path), force="mesh")
            if isinstance(comp_mesh, trimesh.Scene):
                comp_mesh = comp_mesh.dump(concatenate=True)

            vertices = np.asarray(comp_mesh.vertices)
            if len(vertices) == 0:
                continue

            # Scale component vertices with the whole-phone reference scale factor
            vertices = vertices * scale_factor

            # 3. Build Open3D point cloud & estimate normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))

            save_name = f"{folder.name}_{comp_name}.ply"
            save_path = out_dir / save_name
            o3d.io.write_point_cloud(str(save_path), pcd)

            print(f"[OK] {folder.name:4} -> Saved: {save_name:24} | Vertices: {len(vertices)}")
            processed_count += 1

    print(f"\nCompleted: {processed_count} component point clouds generated in {out_dir.resolve()}.")

if __name__ == "__main__":
    process_manual_dataset()