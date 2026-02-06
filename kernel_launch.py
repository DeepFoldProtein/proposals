import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Monospace"

# 1. 데이터 설정 (BLOOM 제외, AF3만 포함) [cite: 114-119]
labels = [
    "aten::linear",
    "aten::layer_norm",
    "aten::native_layer_norm",
    "aten::sigmoid",
    "aten::silu",
]
x = np.arange(len(labels))

# AF3 호출 횟수 데이터 [cite: 100-103, 106]
# GELUFunction은 AF3 데이터가 없으므로 0으로 설정
af3_counts = [37538, 13225, 12686, 10903, 2601]

# 2. 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 6))

# AF3 데이터 막대 생성 (BLOOM 데이터는 제외함)
rects = ax.bar(x, af3_counts, color="#0066FF", edgecolor="black", width=0.5)

# 3. 로그 스케일 설정 및 범위 지정
ax.set_yscale("log")
ax.set_ylim(1, 10**5)

# 4. 막대 위에 수치 라벨 추가 [cite: 100-103, 106]
for rect in rects:
    height = rect.get_height()
    if height > 0:
        ax.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

# 5. 스타일 및 레이블 설정 [cite: 98, 114-119]
ax.set_ylabel("Call Count (Log Scale)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
ax.grid(True, axis="y", linestyle="-", alpha=0.3)

# 6. 범례 및 레이아웃
plt.tight_layout()

# 7. 결과 출력
plt.savefig("kernel_launch.svg", dpi=300, bbox_inches="tight")
