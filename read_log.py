import re
from pathlib import Path

log_file = Path("scripts/training_recogdrive_agent_feedback_8B.txt")  # đổi thành file của bạn

with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Tìm toàn bộ block Teacher
pattern = re.compile(
    r"\[Teacher Trigger\](.*?)"
    r"\[Teacher Feedback\](.*?)"
    r"\[Pass2\](.*?)"
    r"\[Refinement\](.*?)(?=\n\[GRPO Group\]|\Z)",
    re.S,
)

matches = pattern.findall(text)

print(f"\nTotal Teacher Calls: {len(matches)}\n")

for idx, (trigger_block, feedback_block, pass2_block, refine_block) in enumerate(matches, 1):

    step = re.search(r"step=(\d+)", trigger_block)
    reward_before_trigger = re.search(r"reward_before=([-\d.]+)", trigger_block)

    latency = re.search(r"latency=([-\d.]+)s", feedback_block)
    chars = re.search(r"chars=(\d+)", feedback_block)

    preview = re.search(r'preview="(.*?)"', feedback_block, re.S)
    preview_text = preview.group(1).replace("\\n", "\n") if preview else ""

    reward_before = re.search(r"reward_before=([-\d.]+)", pass2_block)
    reward_after = re.search(r"reward_after=([-\d.]+)", pass2_block)

    delta_reward = re.search(r"delta_reward=([+\-\d.]+)", refine_block)

    print("=" * 80)
    print(f"Teacher Call #{idx}")
    print("=" * 80)

    if step:
        print(f"Step            : {step.group(1)}")

    if reward_before_trigger:
        print(f"Trigger Reward  : {reward_before_trigger.group(1)}")

    if latency:
        print(f"Latency         : {latency.group(1)} s")

    if chars:
        print(f"Feedback Length : {chars.group(1)} chars")

    print("\nTeacher Feedback:")
    print("-" * 80)
    print(preview_text)
    print("-" * 80)

    if reward_before and reward_after:
        print(f"Reward Before   : {reward_before.group(1)}")
        print(f"Reward After    : {reward_after.group(1)}")

    if delta_reward:
        print(f"Delta Reward    : {delta_reward.group(1)}")

    print()