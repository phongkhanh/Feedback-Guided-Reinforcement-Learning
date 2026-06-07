import json
from pathlib import Path

# Folder containing image/json pairs
dataset_dir = Path("fail_analysis")

# Large jsonl metadata file
metadata_jsonl = "scripts/dataset_fail_2B_167_all_new.jsonl"

# ------------------------------------------------------------------
# Build lookup table from metadata
# ------------------------------------------------------------------
meta_lookup = {}

with open(metadata_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        meta_lookup[item["token"]] = {
            "traj_bad": item["traj_bad"],
            "traj_GT": item["traj_GT"],
        }

print(f"Loaded {len(meta_lookup)} metadata entries")

# ------------------------------------------------------------------
# Update each json file in dataset folder
# ------------------------------------------------------------------
updated = 0
missing = 0

for json_file in dataset_dir.glob("*.json"):

    with open(json_file, "r", encoding="utf-8") as f:
        sample = json.load(f)

    token = sample.get("token")

    if token not in meta_lookup:
        print(f"[MISSING] {token}")
        missing += 1
        continue

    sample["traj_bad"] = meta_lookup[token]["traj_bad"]
    sample["traj_GT"] = meta_lookup[token]["traj_GT"]

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            sample,
            f,
            indent=2,
            ensure_ascii=False,
        )

    updated += 1

print(f"\nUpdated : {updated}")
print(f"Missing : {missing}")