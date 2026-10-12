import json
import open3d as o3d
import numpy as np
from pathlib import Path

def visualize_grasp(ply_path, json_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pcd.paint_uniform_color([0.7, 0.7, 0.7]) 

    with open(json_path, "r") as f:
        data = json.load(f)

    geometries = [pcd]
    grasps = data.get("grasps", [])
    
    print(f"\n{'='*72}")
    print(f" MICRO-PHYSICS SCORECARD: {ply_path.name}")
    print(f"{'='*72}")
    print(f"Rank | Final Score | Peeling | Anti-Wedge | Slip Resist | Safety Bite")
    print(f"-----|-------------|---------|------------|-------------|------------")

    for idx, g in enumerate(grasps):
        T = np.array(g["grasp_pose_matrix"])
        score = g.get("final_score", 0.0)
        metrics = g.get("metrics", {})
        
        print(f" #{idx:<3} |    {score:.3f}    |  {metrics.get('peeling_score', 0.0):.3f}  |    {metrics.get('anti_wedge_score', 0.0):.3f}   |    {metrics.get('slip_resistance', 0.0):.3f}    |   {metrics.get('safety_margin', 0.0):.3f}")

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.012, origin=[0, 0, 0])
        frame.transform(T)
        geometries.append(frame)
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        if idx == 0:
            sphere.paint_uniform_color([0.0, 1.0, 0.0]) # Green for Best
        else:
            ratio = idx / len(grasps)
            sphere.paint_uniform_color([1.0, 1.0 - ratio, 0.0])
            
        sphere.transform(T)
        geometries.append(sphere)

    print(f"{'='*72}\n")
    print("Visualizer opened! Look for the GREEN sphere to see the winning grasp.")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Grasp Viewer - {ply_path.stem} | GREEN = Best Score",
        width=1200,
        height=800
    )

if __name__ == "__main__":
    ply_files = list(Path("output/components").glob("*.ply"))
    
    if ply_files:
        test_ply = ply_files[1] 
        test_json = Path("output/grasps") / f"{test_ply.stem}_grasps.json"
        
        if test_json.exists():
            visualize_grasp(test_ply, test_json)
        else:
            print(f"Grasp JSON not found for {test_ply.name}. Run generate_grasps.py first.")
    else:
        print("No .ply components found in output/components/. Run process_manual_objs.py first.")