import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Monospace"

# -----------------------------
# Assumptions (H200, 256 GPUs)
# -----------------------------

# Baseline step time at seq_len = 384 (seconds per step)
BASE_SEQ = 384
BASE_STEP_TIME = 14.82  # seconds/step (MegaFold extrapolated estimate)

# Training stages: (name, steps, sequence_length)
stages = [
    ("Stage 0", 74250, 384),
    ("Stage 1", 1750, 640),
    ("Stage 2", 250, 768),
    ("Stage 3", 1750, 768),
]


def step_time(seq_len):
    """
    Estimate step time using cubic scaling:
        t(N) = t_base * (N / N_base)^3
    """
    return BASE_STEP_TIME * (seq_len / BASE_SEQ) ** 3


stage_names = []
stage_hours = []
detailed_info = []

# Compute times
for name, steps, seq_len in stages:
    t_step = step_time(seq_len)
    total_seconds = steps * t_step
    total_hours = total_seconds / 3600.0

    stage_names.append(name)
    stage_hours.append(total_hours)

    detailed_info.append(
        f"{name}: seq={seq_len}, step_time={t_step:.2f}s, total={total_hours:.2f}h"
    )

# Print detailed numbers to console
print("Estimated training time per stage (H200 x256, cubic scaling):")
for line in detailed_info:
    print(line)

print(
    f"\nTotal training time: {sum(stage_hours):.2f} hours "
    f"({sum(stage_hours) / 24:.2f} days)"
)

# -----------------------------
# Visualization
# -----------------------------
plt.figure()
# Colors for each stage
colors = ["#0066FF", "#33CC66", "#FF0066", "#FFAA00"]
plt.bar(stage_names, stage_hours, color=colors, edgecolor="black")

plt.title("Estimated Training Time per Stage (H200 x256)")
plt.ylabel("Time (hours)")
plt.xlabel("Training Stage")

# Annotate bars
for i, v in enumerate(stage_hours):
    plt.text(i, v, f"{v:.1f}h", ha="center", va="bottom")


# Caption for day breakdown
stage_days = [h / 24 for h in stage_hours]
day_sum_str = " + ".join([f"{d:.1f}" for d in stage_days])
total_days = sum(stage_days)
caption_text = f"Total Days: {day_sum_str} = {total_days:.1f} days"

plt.figtext(
    0.5,
    -0.05,
    caption_text,
    ha="center",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, edgecolor="gray"),
)

plt.tight_layout()
plt.savefig("train.svg", dpi=300, bbox_inches="tight")
