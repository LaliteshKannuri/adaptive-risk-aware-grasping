import json
from pathlib import Path

def build_master_dataset():
    grasp_dir = Path("output/grasps")
    json_files = list(grasp_dir.glob("*_grasps.json"))

    master_manifest = {
        "metadata": {
            "total_components": len(json_files),
            "target_robot": "Franka Emika Panda",
            "gripper_max_width_m": 0.08,
            "coordinate_convention": "SE(3) Matrix (+Z approach, +X closing)"
        },
        "components": []
    }

    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
            master_manifest["components"].append(data)

    out_file = Path("output/master_phone_grasps.json")
    with open(out_file, "w") as f:
        json.dump(master_manifest, f, indent=2)

    print(f"Master manifest compiled: {out_file.resolve()}")
    print(f"Successfully aggregated {len(json_files)} component grasp sets.")

if __name__ == "__main__":
    build_master_dataset()