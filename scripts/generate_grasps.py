import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path

MAX_GRIPPER_WIDTH = 0.08  # 80 mm max Franka Hand stroke

def compute_component_grasps(ply_path, num_samples=6):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    
    if len(points) < 50:
        print(f"  └─ Skipped: Not enough points ({len(points)})")
        return []
        
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)
    normals = np.asarray(pcd.normals)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # 2D PCA (TOP-DOWN OVERRIDE)
    points_2d = points[:, :2]
    centroid_2d = np.mean(points_2d, axis=0)
    centered_2d = points_2d - centroid_2d
    cov_2d = np.cov(centered_2d, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    
    order = np.argsort(eigenvalues)
    minor_axis_2d = eigenvectors[:, order[0]]
    major_axis_2d = eigenvectors[:, order[1]]
    
    # Orthogonal Axes Alignment
    z_approach = np.array([0.0, 0.0, -1.0])
    x_closing = np.array([minor_axis_2d[0], minor_axis_2d[1], 0.0])
    y_binormal = np.cross(z_approach, x_closing)
    rot_matrix = np.column_stack((x_closing, y_binormal, z_approach))
    
    # 3D Centroid snapping
    z_max = np.max(points[:, 2])
    centroid_3d = np.array([centroid_2d[0], centroid_2d[1], z_max])
    primary_axis_3d = np.array([major_axis_2d[0], major_axis_2d[1], 0.0])
    
    # Width Validation
    proj_secondary = centered_2d @ minor_axis_2d
    comp_width_m = float(np.ptp(proj_secondary))
    
    if comp_width_m > MAX_GRIPPER_WIDTH:
        print(f"  └─ Skipped: Width ({comp_width_m*1000:.1f} mm) exceeds Franka maximum (80 mm)")
        return []
        
    grasps = []
    proj_primary = centered_2d @ major_axis_2d
    max_offset = np.max(np.abs(proj_primary)) + 1e-6
    
    offsets = np.linspace(np.percentile(proj_primary, 10), np.percentile(proj_primary, 90), num_samples)
    
    # KNN parameters for the contact patch
    k_neighbors = min(30, len(points))
    safe_margin = 0.012  # 12 mm safe bite depth
    
    for idx, offset in enumerate(offsets):
        grasp_center = centroid_3d + (offset * primary_axis_3d)
        
        # ==========================================
        # MICRO-PHYSICS SCORING LOGIC
        # ==========================================
        
        # 1. Edge-Slip Safety Margin & Peeling
        dist_to_edge = max_offset - abs(offset)
        s_safety = np.clip(dist_to_edge / safe_margin, 0.0, 1.0)
        s_peel = (abs(offset) / max_offset) * s_safety  # Peeling is gated by safety
        
        # 2. Anti-Wedge Alignment
        s_wedge = 1.0
        
        # 3. Contact Patch Flatness & Slip Resistance
        cp_left = grasp_center + (x_closing * (comp_width_m / 2.0))
        cp_right = grasp_center - (x_closing * (comp_width_m / 2.0))
        
        _, idx_left, _ = pcd_tree.search_knn_vector_3d(cp_left, k_neighbors)
        _, idx_right, _ = pcd_tree.search_knn_vector_3d(cp_right, k_neighbors)
        
        # Extract normal patches and calculate mean normal vector
        n_left_mean = np.mean(normals[idx_left], axis=0)
        n_right_mean = np.mean(normals[idx_right], axis=0)
        
        # Flatness magnitude (1.0 means perfectly flat, <1.0 means curved/bumpy)
        flatness_left = np.linalg.norm(n_left_mean)
        flatness_right = np.linalg.norm(n_right_mean)
        
        if flatness_left > 0 and flatness_right > 0:
            n_left_unit = n_left_mean / flatness_left
            n_right_unit = n_right_mean / flatness_right
            dot_prod = np.clip(np.dot(n_left_unit, n_right_unit), -1.0, 1.0)
            antipodal_score = 0.5 * (1.0 - dot_prod)
        else:
            antipodal_score = 0.0
            
        # Slip score merges flat surface geometry with opposing forces
        s_slip = antipodal_score * (flatness_left * flatness_right)
        
        # 4. Final Unified Score
        w_peel, w_wedge, w_slip = 0.25, 0.30, 0.45
        q_final = (w_peel * s_peel) + (w_wedge * s_wedge) + (w_slip * s_slip)
        
        # ==========================================
        
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = grasp_center
        
        grasps.append({
            "grasp_id": idx,
            "final_score": round(float(q_final), 3),
            "metrics": {
                "peeling_score": round(float(s_peel), 3),
                "anti_wedge_score": round(float(s_wedge), 3),
                "slip_resistance": round(float(s_slip), 3),
                "safety_margin": round(float(s_safety), 3)
            },
            "opening_width_m": round(min(comp_width_m + 0.005, MAX_GRIPPER_WIDTH), 4),
            "opening_width_mm": round((min(comp_width_m + 0.005, MAX_GRIPPER_WIDTH)) * 1000, 1),
            "grasp_pose_matrix": T.tolist(),
            "position_xyz": [round(float(v), 4) for v in grasp_center],
            "approach_vector": [round(float(v), 4) for v in z_approach],
            "retreat_distance_m": 0.05
        })
        
    grasps.sort(key=lambda x: x["final_score"], reverse=True)
    for i, g in enumerate(grasps):
        g["grasp_id"] = i
        
    return grasps

def generate_all_grasps():
    comp_dir = Path("output/components")
    out_dir = Path("output/grasps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ply_files = list(comp_dir.glob("*.ply"))
    print(f"Generating Micro-Physics grasps for {len(ply_files)} components...\n")
    
    valid_count = 0
    for ply_path in ply_files:
        stem = ply_path.stem
        grasps = compute_component_grasps(ply_path)
        
        if grasps:
            out_file = out_dir / f"{stem}_grasps.json"
            with open(out_file, "w") as f:
                json.dump({
                    "component_id": stem,
                    "gripper": "franka_panda_hand",
                    "max_gripper_width_m": MAX_GRIPPER_WIDTH,
                    "grasps": grasps
                }, f, indent=2)
            
            top_grasp = grasps[0]
            print(f"[OK] {stem:25} -> Width: {top_grasp['opening_width_mm']:5.1f} mm | Score: {top_grasp['final_score']:.3f}")
            valid_count += 1

    print(f"\nGenerated physics-aware grasps for {valid_count}/{len(ply_files)} components.")

if __name__ == "__main__":
    generate_all_grasps()