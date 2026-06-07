import os
import json
import pickle
import cv2
import numpy as np
from pprint import pprint

# =========================================================
# PATH CONFIG
# =========================================================
jsonl_path = "scripts/dataset_fail_2B_167_all_new.jsonl"

logs_root = "/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/navsim_logs/trainval"

sensor_root = "/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/sensor_blobs/trainval"

output_dir = "fail_analysis"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# FORMAT TRAJECTORY
# =========================================================
def format_traj(traj):
    return "[" + ", ".join(
        [f"[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]" for p in traj]
    ) + "]"


# =========================================================
# SYMBOLIC TRAJECTORY ANALYSIS
# =========================================================
def analyze_trajectory_geometry(traj, traj_gt):

    traj = np.array(traj)
    traj_gt = np.array(traj_gt)

    y = traj[:, 1]

    # =====================================================
    # oscillation
    # =====================================================
    sign_flips = int(np.sum(np.diff(np.sign(y)) != 0))

    oscillation = "YES" if sign_flips >= 2 else "NO"

    # =====================================================
    # lateral deviation
    # =====================================================
    lateral_dev = np.abs(y - traj_gt[:, 1])

    max_lateral_dev = float(np.max(lateral_dev))

    mean_lateral_dev = float(np.mean(lateral_dev))

    # =====================================================
    # endpoint error
    # =====================================================
    endpoint_error = float(
        np.linalg.norm(
            traj[-1, :2] - traj_gt[-1, :2]
        )
    )

    # =====================================================
    # heading instability
    # =====================================================
    heading_std = float(np.std(traj[:, 2]))

    # =====================================================
    # monotonicity
    # =====================================================
    monotonic_x = bool(np.all(np.diff(traj[:, 0]) >= -0.5))

    return {
        "oscillation": oscillation,
        "sign_flips": sign_flips,
        "max_lateral_dev": round(max_lateral_dev, 2),
        "mean_lateral_dev": round(mean_lateral_dev, 2),
        "endpoint_error": round(endpoint_error, 2),
        "heading_std": round(heading_std, 3),
        "monotonic_x": monotonic_x,
    }


# =========================================================
# SCENE RELATION EXTRACTION
# =========================================================
def extract_scene_relations(sample):

    anns = sample.get("anns", {})

    gt_boxes = anns.get("gt_boxes", [])
    gt_names = anns.get("gt_names", [])

    if len(gt_boxes) == 0:
        return None

    nearest_dist = 9999
    nearest_obj = None

    for box, name in zip(gt_boxes, gt_names):

        try:
            x = float(box[0])
            y = float(box[1])

            dist = np.sqrt(x**2 + y**2)

            if dist < nearest_dist:

                nearest_dist = dist

                side = "center"

                if y > 1:
                    side = "left"
                elif y < -1:
                    side = "right"

                nearest_obj = {
                    "name": str(name),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "dist": round(float(dist), 2),
                    "side": side,
                }

        except Exception:
            continue

    return nearest_obj


# =========================================================
# BUILD PROMPT
# =========================================================
def build_teacher_prompt(item, sample, ego):

    # =====================================================
    # command
    # =====================================================
    cmd_map = {
        0: "GO STRAIGHT",
        1: "TURN LEFT",
        2: "TURN RIGHT",
        3: "STOP",
    }

    cmd_idx = None

    driving_command = sample.get("driving_command", None)

    if isinstance(driving_command, (list, np.ndarray)):

        try:
            cmd_idx = int(np.argmax(driving_command))
        except Exception:
            cmd_idx = None

    command = cmd_map.get(cmd_idx, "UNKNOWN")

    # =====================================================
    # ego state
    # =====================================================
    if isinstance(ego, list) and len(ego) >= 4:

        speed = ego[0]
        acc_x = ego[1]
        acc_y = ego[2]
        yaw = ego[3]

    else:

        speed = 0.0
        acc_x = 0.0
        acc_y = 0.0
        yaw = 0.0

    acc_desc = "stable"

    if acc_x < -0.1:
        acc_desc = "decelerating"
    elif acc_x > 0.1:
        acc_desc = "accelerating"

    # =====================================================
    # trajectories
    # =====================================================
    traj_gt = item.get("traj_GT", None)
    traj_bad = item.get("traj_bad", None)

    # =====================================================
    # metrics
    # =====================================================
    worst_metrics = item.get("worst_metrics", {})

    collision = worst_metrics.get(
        "no_at_fault_collisions",
        1.0
    )

    dac = worst_metrics.get(
        "drivable_area_compliance",
        1.0
    )

    progress = worst_metrics.get(
        "ego_progress",
        1.0
    )

    ttc = worst_metrics.get(
        "time_to_collision_within_bound",
        1.0
    )

    comfort = worst_metrics.get(
        "comfort",
        1.0
    )

    # =====================================================
    # symbolic metric text
    # =====================================================
    collision_text = (
        "YES" if collision >= 1.0 else "NO"
    )

    dac_text = (
        "YES" if dac >= 1.0 else "NO"
    )

    ttc_text = (
        "SAFE" if ttc >= 1.0 else "UNSAFE"
    )

    comfort_text = (
        "PASS" if comfort >= 1.0 else "FAIL"
    )

    # =====================================================
    # behavior desc
    # =====================================================
    behavior_desc = (
        "unsafe driving behavior"
        if collision < 1.0 or dac < 1.0 or ttc < 1.0
        else "suboptimal driving behavior"
    )

    # =====================================================
    # symbolic geometry analysis
    # =====================================================
    geometry_info = analyze_trajectory_geometry(
        traj_bad,
        traj_gt
    )

    # =====================================================
    # scene relation
    # =====================================================
    scene_relation = extract_scene_relations(sample)

    scene_relation_text = ""

    if scene_relation is not None:

        scene_relation_text = f"""
Scene Relation Analysis:
- Closest object: {scene_relation['name']}
- Relative position: {scene_relation['dist']:.2f}m away
- Relative direction: {scene_relation['side']}
"""

    # =====================================================
    # prompt
    # =====================================================
    prompt = f"""
You are an expert autonomous driving system specializing in trajectory planning and refinement.

Scene: front-view driving.

Navigation Command: {command}

Ego State:
- Speed: {speed:.2f} m/s
- Motion: vehicle is {acc_desc}
- Lateral acceleration: {acc_y:.2f} m/s²
- Yaw rate: {yaw:.2f} rad/s

PDM Metrics:
- Collision-Free: {collision_text}
- Drivable Area Compliance: {dac_text}
- Progress Score: {progress:.2f}
- TTC Safety: {ttc_text}
- Comfort: {comfort_text}

Trajectory Geometry Analysis:
- Lateral Oscillation: {geometry_info['oscillation']}
- Oscillation Count: {geometry_info['sign_flips']}
- Maximum Lateral Deviation: {geometry_info['max_lateral_dev']} m
- Mean Lateral Deviation: {geometry_info['mean_lateral_dev']} m
- Endpoint Error: {geometry_info['endpoint_error']} m
- Heading Instability: {geometry_info['heading_std']}
- Monotonic Forward Progress: {geometry_info['monotonic_x']}

{scene_relation_text}

Trajectories:
- Ground Truth: {format_traj(traj_gt)}
- Fault Trajectory: {format_traj(traj_bad)}

The fault trajectory deviates from the ground truth and results in {behavior_desc}.

Task:
Analyze why the fault trajectory fails compared to the ground truth trajectory.

Focus on:
- trajectory instability
- oscillation or excessive lateral deviation
- route inconsistency
- unsafe or inefficient driving behavior
- metric-related failures

Provide concise refinement guidance to improve trajectory quality, safety, and stability.

Rules:
- Do not hallucinate scene details that are not provided.
- Focus on geometric and semantic trajectory behavior.
- Avoid overly generic explanations.
- Explicitly describe trajectory motion patterns when relevant.

Output format:

[Trajectory Issue]
<brief summary of the main trajectory failure>

[Reasoning]
<explain how the fault trajectory differs from the ground truth and why it causes metric failures>

[Refinement]
<describe how the trajectory should be improved for safer and smoother driving>
"""

    return prompt.strip()


if __name__ == "__main__":
    # =====================================================
    # BUILD TOKEN MAP
    # =====================================================
    print("Building token index...")

    token_map = {}

    for fname in os.listdir(logs_root):

        if not fname.endswith(".pkl"):
            continue

        with open(
            os.path.join(logs_root, fname),
            "rb"
        ) as f:

            data = pickle.load(f)

        for sample in data:
            token_map[sample["token"]] = sample

    print("Total tokens:", len(token_map))


    # =====================================================
    # READ JSONL
    # =====================================================
    with open(jsonl_path, "r") as f:
        lines = f.readlines()


    # =====================================================
    # LOOP
    # =====================================================
    for line in lines:

        item = json.loads(line.strip())

        token = item["token"]

        if item.get("type") == "easy":
            continue

        if token not in token_map:
            continue

        sample = token_map[token]

        # =================================================
        # image
        # =================================================
        cam_path = sample["cams"]["CAM_F0"]["data_path"]

        img_path = os.path.join(sensor_root, cam_path)

        img_saved = False

        if os.path.exists(img_path):

            img = cv2.imread(img_path)

            out_img_path = os.path.join(output_dir, f"{token}.jpg")
            cv2.imwrite(out_img_path, img)
            img_saved = True

        # =================================================
        # ego + prompt
        # =================================================
        ego = sample.get("ego_dynamic_state", {})

        prompt = build_teacher_prompt(item, sample, ego)

        out_json_path = os.path.join(output_dir, f"{token}.json")

        with open(out_json_path, "w") as f:
            json.dump({"token": token, "type": item["type"], "prompt": prompt}, f, indent=2)

        print(f"[{'OK' if img_saved else 'NO IMG'}] {token}")