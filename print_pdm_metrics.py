import pandas as pd

CSV_PATH = "navsim_workspace/exp/recogdrive_agent_eval_feedback_sup/2026.06.06.18.40.05/2026.06.06.18.56.10.csv"

def print_metrics(csv_path: str):
    df = pd.read_csv(csv_path)
    valid_df = df[df["valid"] == True]

    metrics = [
        ("NC↑ ", "no_at_fault_collisions"),
        ("DAC↑", "drivable_area_compliance"),
        ("TTC↑", "time_to_collision_within_bound"),
        ("CF↑ ", "comfort"),
        ("EP↑ ", "ego_progress"),
        ("DDC↑", "driving_direction_compliance"),
        ("PDMS↑", "score"),
    ]

    print(f"\nResults from: {csv_path}")
    print(f"Scenarios: {len(valid_df)} valid / {len(df)} total")
    print("-" * 45)
    for label, col in metrics:
        if col in valid_df.columns:
            print(f"  {label:<6} {valid_df[col].mean():.4f}")
    print("-" * 45)


if __name__ == "__main__":
    print_metrics(CSV_PATH)
