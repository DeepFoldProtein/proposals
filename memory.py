import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Monospace"

# 1. 데이터 설정 (시각적 근사치 반영) [cite: 34-39]
labels = ["128", "256", "384", "512", "640", "768"]
x = np.arange(len(labels))  # 라벨 위치
width = 0.25  # 막대 너비

# 모델별 메모리 사용량 (GB) [cite: 22-23, 25-30]
# 0으로 표시된 부분은 OOM(Out of Memory) 지점입니다.
y_eager = [22, 40, 61, 95, 0, 0]
y_inductor = [22, 40, 61, 95, 0, 0]
y_optimized = [22, 37, 52, 78, 109, 139]

# 2. 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 6))

# 막대 배치
rects1 = ax.bar(
    x - width,
    y_eager,
    width,
    label="PyTorch (Eager)",
    color="#0066FF",
    edgecolor="black",
)
rects2 = ax.bar(
    x, y_inductor, width, label="PyTorch (Inductor)", color="#33CC66", edgecolor="black"
)
rects3 = ax.bar(
    x + width, y_optimized, width, label="Optimized", color="#FF0066", edgecolor="black"
)


# 3. OOM(Out of Memory) 표시 (640, 768 지점) [cite: 31, 32, 38, 39]
for i in [4, 5]:  # 640과 768 인덱스
    # X 표시
    ax.text(i - width, 2, "X", color="red", ha="center", fontweight="bold", fontsize=14)
    ax.text(i, 2, "X", color="red", ha="center", fontweight="bold", fontsize=14)
    # OOM 텍스트
    ax.text(i - width / 2, 8, "OOM", color="darkred", ha="center", fontsize=10)

# 4. 스타일 및 레이블 설정 [cite: 26, 40]
ax.set_ylabel("Peak Memory Allocated (GB)", fontsize=12, fontweight="bold")
ax.set_xlabel("Sequence Length (Checkpointed)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
ax.set_ylim(0, 155)
ax.grid(axis="y", linestyle="-", alpha=0.3)

# 5. 범례 및 레이아웃 [cite: 24]
ax.legend(loc="lower left", fontsize=12)
plt.tight_layout()

# 6. H200 Limit 표시 (좌측 상단 화살표 및 텍스트) [cite: 21, 22]
limit_val = 141.4  # 일반적인 H200 메모리 한계치
ax.axhline(y=limit_val, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
ax.annotate(
    "H200 Limit",
    xy=(0, limit_val),
    xytext=(-0.5, limit_val + 5),
    arrowprops=dict(arrowstyle="->", color="black"),
)


# 7. 결과 출력
plt.savefig("memory.svg", dpi=300, bbox_inches="tight")
