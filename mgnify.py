#!/usr/bin/env python3
"""
Rough AF2 inference time visualization for MGnify length histogram (H200 scaling).

Assumptions:
- Each length bin is represented by its midpoint length.
- A100 inference times are linearly interpolated between given residue-time anchors.
- Time scales inversely with peak BF16/FP16 Tensor Core FLOPS (very rough).
- Ideal parallel efficiency across GPUs (no I/O, scheduling, or memory bottlenecks).

Outputs (English):
- Figure 1: Sequence count by length bin (log scale) + per-sequence time on H200.
- Figure 2: Wall-clock time contribution by bin on a 128x H200 cluster + cumulative.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Monospace"


# ----------------------------
# Input: MGnify length histogram (provided)
# ----------------------------
bins = [
    (200, 299, 106_232_237),
    (300, 399, 43_382_930),
    (400, 499, 21_212_845),
    (500, 599, 10_100_177),
    (600, 699, 5_679_747),
    (700, 799, 3_390_077),
    (800, 899, 2_195_438),
    (900, 999, 1_330_801),
    (1000, 1099, 969_374),
]

# Midpoint residue length for each bin
mid_lengths = np.array([(lo + hi) / 2.0 for lo, hi, _ in bins], dtype=float)
counts = np.array([c for _, _, c in bins], dtype=float)

# ----------------------------
# Input: A100 AF2 inference times (anchors)
# ----------------------------
# (residues, seconds)
a100_anchors = np.array(
    [
        (100, 4.9),
        (200, 7.7),
        (300, 13.0),
        (400, 18.0),
        (500, 29.0),
        (600, 36.0),
        (700, 53.0),
        (800, 60.0),
        (900, 91.0),
        (1000, 96.0),
        (1100, 140.0),
        (1500, 280.0),
        (2000, 450.0),
    ],
    dtype=float,
)

anchor_x = a100_anchors[:, 0]
anchor_y = a100_anchors[:, 1]


def interp_time_seconds(residues: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation of A100 seconds vs residues."""
    # np.interp clamps outside range; here mid_lengths are within anchors (200~1099)
    return np.interp(residues, anchor_x, anchor_y)


# A100 per-sequence times for each bin midpoint (seconds)
a100_sec_per_seq = interp_time_seconds(mid_lengths)

# ----------------------------
# FLOPS scaling (very rough)
# ----------------------------
# Peak BF16/FP16 Tensor Core FLOPS (dense) used for scaling
A100_TFLOPS = 312.0
H200_TFLOPS = 1979.0
speedup = H200_TFLOPS / A100_TFLOPS  # >1 means H200 is faster

# H200 per-sequence time (seconds)
h200_sec_per_seq = a100_sec_per_seq / speedup

# ----------------------------
# Aggregate time calculations
# ----------------------------
# Total GPU-seconds on H200 if run serially on 1 GPU
h200_total_gpu_seconds_per_bin = counts * h200_sec_per_seq
h200_total_gpu_hours_per_bin = h200_total_gpu_seconds_per_bin / 3600.0

# Cluster wall-clock time on N GPUs (ideal)
N_GPUS = 128
wall_hours_per_bin = h200_total_gpu_hours_per_bin / N_GPUS
wall_days_per_bin = wall_hours_per_bin / 24.0

# Totals
total_sequences = counts.sum()
total_wall_days = wall_days_per_bin.sum()
total_wall_hours = wall_hours_per_bin.sum()
total_h200_gpu_hours = h200_total_gpu_hours_per_bin.sum()

# ----------------------------
# Labels for plots
# ----------------------------
bin_labels = [f"{lo}-{hi}" for lo, hi, _ in bins]
x = np.arange(len(bins))

# ----------------------------
# Plot 1: Counts + per-sequence time
# ----------------------------
fig1, ax1 = plt.subplots(figsize=(10.5, 5.5))
bars = ax1.bar(x, counts, width=0.75, color="#0066FF", edgecolor="black")
ax1.set_yscale("log")
ax1.set_xlabel("Residue length bin")
ax1.set_ylabel("Sequence count (log scale)")
ax1.set_xticks(x, bin_labels, rotation=35, ha="right")
ax1.set_title(
    "MGnify sequence length histogram + AF2 per-sequence inference time (H200-scaled)"
)

ax2 = ax1.twinx()
ax2.plot(x, h200_sec_per_seq, marker="o", color="#FF0066", linewidth=2)
ax2.set_ylabel("Per-sequence inference time on H200 (seconds)")

# Add a small summary box
summary = (
    f"Total sequences: {total_sequences:,.0f}\n"
    f"H200 GPU-hours (serial): {total_h200_gpu_hours:,.0f}\n"
    f"Wall-clock on {N_GPUS}×H200 (ideal): {total_wall_days:,.1f} days"
)
ax1.text(
    0.02,
    0.02,
    summary,
    transform=ax1.transAxes,
    va="bottom",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig1.tight_layout()

# ----------------------------
# Plot 2: Wall-clock contribution by bin + cumulative
# ----------------------------
fig2, ax3 = plt.subplots(figsize=(10.5, 5.5))
ax3.bar(x, wall_days_per_bin, width=0.75, color="#0066FF", edgecolor="black")
ax3.set_xlabel("Residue length bin")
ax3.set_ylabel(f"Wall-clock time contribution (days) on {N_GPUS}×H200")
ax3.set_xticks(x, bin_labels, rotation=35, ha="right")
ax3.set_title(f"Wall-clock time by length bin (ideal scaling to {N_GPUS} H200 GPUs)")

cum_days = np.cumsum(wall_days_per_bin)
ax4 = ax3.twinx()
ax4.plot(x, cum_days, marker="o", color="#FF0066", linewidth=2)
ax4.set_ylabel("Cumulative wall-clock time (days)")

# Annotate total
ax3.axhline(0, linewidth=1)
ax3.text(
    0.02,
    0.98,
    f"Total wall-clock (ideal): {total_wall_days:,.1f} days ({total_wall_hours:,.0f} hours)",
    transform=ax3.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig2.tight_layout()

# ----------------------------
# Show
# ----------------------------
plt.savefig("mgnify.svg", dpi=300, bbox_inches="tight")
