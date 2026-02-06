import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Monospace"

# 1. 데이터 설정 (시각적 근사치) [cite: 14-20]
x = [128, 256, 384, 512, 640, 768]

# 각 모델별 실행 시간 (s) 추정 데이터 [cite: 1, 4, 6]
y_eager = [6.6, 8.1, 10.5, 14.7, None, None]  # PyTorch (Eager Mode)
y_inductor = [6.7, 8.2, 10.9, 15.0, None, None]  # PyTorch (Inductor)
# y_optimized = [3.7, 4.6, 7.3, 8.5, 13.2, 17.8]  # MegaFold
y_optimized = [3.7, 4.6, 6.8, 8.5, 13.2, 17.8]  # MegaFold

# 2. 그래프 스타일 설정
plt.figure(figsize=(10, 6))
plt.grid(True, axis="y", linestyle="-", alpha=0.3)  # 가로 그리드 설정

# 3. 각 라인 그리기
plt.plot(
    x[:4],
    y_eager[:4],
    marker="o",
    markersize=8,
    linewidth=2,
    label="PyTorch (Eager Mode)",
    color="#0066FF",
)
plt.plot(
    x[:4],
    y_inductor[:4],
    marker="o",
    markersize=8,
    linewidth=2,
    label="PyTorch (Inductor)",
    color="#33CC66",
)
plt.plot(
    x,
    y_optimized,
    marker="o",
    markersize=8,
    linewidth=2,
    label="Optimized",
    color="#FF0066",
)

# 4. 빨간색 X 마커 추가 (Out of Memory 또는 실패 지점 표현)
# plt.scatter([640, 768], [0.3, 0.3], marker="x", color="red", s=100, linewidths=3)

# 5. 축 및 레이블 설정 [cite: 1, 20]
plt.xlabel("Sequence Length", fontsize=12, fontweight="bold")
plt.ylabel("Execution Time (s)", fontsize=12, fontweight="bold")
plt.xticks(x)
plt.yticks([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
plt.ylim(0, 21)

# 6. 범례 및 레이아웃
plt.legend(loc="upper left", fontsize=12)
plt.tight_layout()

# 7. 결과 출력
plt.savefig("speed.svg", dpi=300, bbox_inches="tight")
