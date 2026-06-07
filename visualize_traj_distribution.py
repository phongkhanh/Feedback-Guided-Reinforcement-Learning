import os
import json
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== CONFIG =====
jsonl_path = "scripts/dataset_fail_2B_167_all.jsonl"
logs_root = "/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/navsim_logs/trainval"
sensor_root = "/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/sensor_blobs/trainval"
output_dir = "./vis_output"

os.makedirs(output_dir, exist_ok=True)

# ===== BUILD TOKEN MAP =====
print("Building token index...")
token_map = {}

for fname in os.listdir(logs_root):
    if not fname.endswith(".pkl"):
        continue
    with open(os.path.join(logs_root, fname), "rb") as f:
        data = pickle.load(f)
    for sample in data:
        token_map[sample["token"]] = sample

print("Total tokens:", len(token_map))


# ===== DRAW =====
def draw_traj(ax, traj, color, alpha=1.0, linewidth=2, label=None):
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    ax.plot(xs, ys, marker='o', color=color, alpha=alpha, linewidth=linewidth, label=label)


# ===== CHOOSE TYPE TO VISUALIZE =====
VIS_TYPES = ["easy", "ambiguous", "difficult"]   # chỉnh ở đây

# ===== MAIN =====
with open(jsonl_path, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(tqdm(lines)):
    try:
        item = json.loads(line.strip())
    except:
        continue

    if item["type"] not in VIS_TYPES:
        continue

    token = item["token"]
    if token not in token_map:
        continue

    sample = token_map[token]

    # ===== LOAD IMAGE =====
    try:
        cam_path = sample["cams"]["CAM_F0"]["data_path"]
        img_path = os.path.join(sensor_root, cam_path)
    except:
        continue

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ===== TRAJECTORIES =====
    traj_all = item.get("traj_all", [])
    rewards = item.get("rewards", [])
    traj_GT = item.get("traj_GT", None)

    best_idx = np.argmax(rewards)
    worst_idx = np.argmin(rewards)

    # ===== FIG =====
    fig = plt.figure(figsize=(14, 6))

    # ================= IMAGE =================
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title(f"{token}\n{item['type']} | mean={item['mean']:.2f} std={item['std']:.2f}")
    ax1.axis("off")

    # ================= BEV =================
    ax2 = plt.subplot(1, 2, 2)
    import matplotlib.cm as cm

    # normalize reward về [0,1]
    r = np.array(rewards)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-6)

    cmap = cm.get_cmap("viridis")  # đẹp, dễ nhìn
    # 🔥 draw ALL traj (mờ)
    # for i, traj in enumerate(traj_all):
    #     draw_traj(ax2, traj, color="black", alpha=0.3, linewidth=1)
    for i, traj in enumerate(traj_all):
        if i == 0:
            draw_traj(ax2, traj, color="black", alpha=0.3, linewidth=1, label="others")
        else:
            draw_traj(ax2, traj, color="black", alpha=0.3, linewidth=1)
    # for i, traj in enumerate(traj_all):
    #     color = cmap(r_norm[i])  # màu theo reward
    #     draw_traj(ax2, traj, color=color, alpha=0.8, linewidth=2)

    # 🔥 draw BEST
    draw_traj(ax2, traj_all[best_idx], color="green", linewidth=3, label=f"best ({rewards[best_idx]:.2f})")

    # 🔥 draw WORST
    draw_traj(ax2, traj_all[worst_idx], color="red", linewidth=3, label=f"worst ({rewards[worst_idx]:.2f})")

    # 🔥 draw GT
    if traj_GT is not None:
        draw_traj(ax2, traj_GT, color="blue", linewidth=2, label="GT")

    # ===== format =====
    ax2.legend()
    ax2.set_title("Trajectory Distribution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.invert_yaxis()
    ax2.grid()

    # ===== TEXT REWARD =====
    reward_text = "\n".join([f"{i}: {r:.2f}" for i, r in enumerate(rewards)])
    ax2.text(1.05, 0.5, reward_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='center')

    # ===== SAVE =====
    save_path = os.path.join(output_dir, f"{item['type']}_{idx}_{token}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

print("Done!")