import os
import json
import time
import trimesh
import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R

Z_OFFSET = np.array([0.0, 0.0, 0.03])  # Lift workspace 3 cm off the ground plane
current_phone_body = None
current_comp_body = None

# 1. Initialize PyBullet
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.resetDebugVisualizerCamera(
    cameraDistance=0.55,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0.0, 0.0, 0.05]
)

# 2. Load Table & Franka Robot
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", [0.0, 0.0, -0.62], [0, 0, 0, 1])

panda_id = p.loadURDF("franka_panda/panda.urdf", [0.0, -0.55, 0.0], useFixedBase=True)
ee_link_index = 11
finger_indices = [9, 10]

def set_gripper_width(width):
    half_width = width / 2.0
    for finger_idx in finger_indices:
        p.setJointMotorControl2(
            panda_id, finger_idx, p.POSITION_CONTROL,
            targetPosition=half_width, force=30
        )

def move_arm_to_pose(target_pos, target_quat, steps=80):
    rest_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Natural elbow configuration
    joint_poses = p.calculateInverseKinematics(
        panda_id, ee_link_index, target_pos, target_quat,
        restPoses=rest_joints,
        maxNumIterations=200,
        residualThreshold=1e-4
    )
    for i in range(7):
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, joint_poses[i], force=120)

    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1.0 / 120.0)

def find_chassis_mesh(comp_id):
    """Finds model_Other.obj or model_whole.obj based on folder ID."""
    folder_num = comp_id.split('_')[0]
    folder_path = Path(f"dataset/{folder_num}")

    other_path = folder_path / "model_Other.obj"
    if other_path.exists():
        return other_path

    whole_path = folder_path / "model_whole.obj"
    if whole_path.exists():
        return whole_path

    return None

def render_context(comp_id):
    """Loads Ghost chassis URDF (visual-only) and component collision block."""
    global current_phone_body, current_comp_body

    if current_phone_body is not None:
        p.removeBody(current_phone_body)
    if current_comp_body is not None:
        p.removeBody(current_comp_body)
    p.removeAllUserDebugItems()

    # 1. Visual-Only Ghost Chassis
    obj_path = find_chassis_mesh(comp_id)
    if obj_path:
        try:
            loaded_geom = trimesh.load(str(obj_path), force="mesh")
            if isinstance(loaded_geom, trimesh.Scene):
                mesh = loaded_geom.dump(concatenate=True)
            else:
                mesh = loaded_geom
            scale_factor = 0.155 / (np.max(mesh.extents) + 1e-8)
        except Exception:
            scale_factor = 1.0

        ghost_urdf = obj_path.with_name(f"{obj_path.stem}_ghost.urdf")
        urdf_content = f"""<?xml version="1.0" ?>
        <robot name="{comp_id}_ghost">
          <link name="base_link">
            <visual>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <mesh filename="{obj_path.name}" scale="{scale_factor} {scale_factor} {scale_factor}"/>
              </geometry>
            </visual>
          </link>
        </robot>
        """
        with open(ghost_urdf, "w") as f:
            f.write(urdf_content)

        current_phone_body = p.loadURDF(str(ghost_urdf), basePosition=Z_OFFSET, useFixedBase=True)

    # 2. Physical Collision Component Block
    ply_path = Path(f"output/components/{comp_id}.ply")
    if ply_path.exists():
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        if len(points) > 0:
            points = points + Z_OFFSET
            min_bounds, max_bounds = np.min(points, axis=0), np.max(points, axis=0)
            center = (min_bounds + max_bounds) / 2.0
            half_extents = (max_bounds - min_bounds) / 2.0
            half_extents[2] = max(half_extents[2], 0.005)

            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.0, 1.0, 1.0, 0.5])
            current_comp_body = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=center
            )

def execute_grasp_sequence(grasp_data, comp_id):
    T = np.array(grasp_data["grasp_pose_matrix"])
    grasp_pos = T[:3, 3] + Z_OFFSET
    rot_matrix = T[:3, :3]
    grasp_quat = R.from_matrix(rot_matrix).as_quat()

    approach_vec = np.array(grasp_data["approach_vector"])
    pre_grasp_pos = grasp_pos - (approach_vec * 0.06)
    opening_width = grasp_data.get("opening_width_m", 0.08)

    print(f"\nExecuting: {comp_id} | Grasp ID: {grasp_data['grasp_id']} | Opening: {opening_width*1000:.1f} mm")

    render_context(comp_id)

    # Grasp Coordinate Frame Display
    p.addUserDebugLine(grasp_pos, grasp_pos + rot_matrix[:, 0] * 0.03, [1, 0, 0], lineWidth=3)
    p.addUserDebugLine(grasp_pos, grasp_pos + rot_matrix[:, 1] * 0.03, [0, 1, 0], lineWidth=3)
    p.addUserDebugLine(grasp_pos, grasp_pos + rot_matrix[:, 2] * 0.04, [0, 0, 1], lineWidth=3)

    set_gripper_width(opening_width)
    move_arm_to_pose(pre_grasp_pos, grasp_quat, steps=70)
    move_arm_to_pose(grasp_pos, grasp_quat, steps=50)

    set_gripper_width(0.0)
    for _ in range(40):
        p.stepSimulation()
        time.sleep(1.0 / 120.0)

    retreat_pos = grasp_pos + np.array([0, 0, 0.15])
    move_arm_to_pose(retreat_pos, grasp_quat, steps=80)
    time.sleep(0.5)

def run_simulation():
    master_file = Path("output/master_phone_grasps.json")
    if not master_file.exists():
        print(f"Missing {master_file}. Run export_master_grasps.py first.")
        return

    with open(master_file, "r") as f:
        master_data = json.load(f)

    components = master_data.get("components", [])
    print(f"Loaded {len(components)} component sets from master dataset.")

    for comp in components:
        if len(comp["grasps"]) > 0:
            execute_grasp_sequence(comp["grasps"][0], comp["component_id"])

    print("\nSimulation complete.")
    time.sleep(2.0)
    p.disconnect()

if __name__ == "__main__":
    run_simulation()