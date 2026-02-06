import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = "Monospace"

# 1. 데이터 설정 [cite: 63-67]
seq_lengths = ["64", "96", "128", "160", "192"]
x = np.arange(len(seq_lengths))

# 각 구성 요소별 메모리 사용량 (GB) [cite: 46, 53, 56, 58, 60, 61]
activation_memory = [17.8, 33.5, 52.7, 77.2, 107.4]
model_weights = [3.1] * 5  # 추정치 (고정)
gradients = [3.1] * 5  # 추정치 (고정)
optimizer_states = [4.08] * 5  # 이미지에 명시된 값

# 2. 그래프 그리기 (누적 막대 그래프)
plt.figure(figsize=(10, 6))

# 하단부터 쌓아 올리기
p1 = plt.bar(
    x,
    activation_memory,
    label="Activation Memory",
    color="#0066FF",
    edgecolor="black",
    width=0.6,
)
p2 = plt.bar(
    x,
    model_weights,
    bottom=activation_memory,
    label="Model Weights",
    color="#33CC66",
    edgecolor="black",
    width=0.6,
)
p3 = plt.bar(
    x,
    gradients,
    bottom=np.array(activation_memory) + np.array(model_weights),
    label="Gradients",
    color="#FF0066",
    edgecolor="black",
    width=0.6,
)
p4 = plt.bar(
    x,
    optimizer_states,
    bottom=np.array(activation_memory) + np.array(model_weights) + np.array(gradients),
    label="Optimizer States",
    color="#FFAA00",
    edgecolor="black",
    width=0.6,
)

# 3. 수치 라벨 추가 [cite: 46, 53, 56, 58, 60, 61]
for i in range(len(x)):
    # Activation Memory 값 표시 (주황색 막대 오른쪽)
    plt.text(
        i,
        activation_memory[i] / 2,
        f"{activation_memory[i]}",
        ha="center",
        va="center",
        fontsize=11,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )
    # Optimizer States 값 표시 (보라색 막대 오른쪽 상단)
    total_height = (
        activation_memory[i] + model_weights[i] + gradients[i] + optimizer_states[i]
    )
    plt.text(i + 0.32, total_height - 2, "4.08", va="center", fontsize=11)

# 4. 스타일 및 레이블 설정 [cite: 41, 45, 47, 48, 49, 50, 54, 59, 62, 68]
plt.ylabel("End-to-End Memory Usage (GB)", fontsize=12, fontweight="bold")
plt.xlabel("Sequence Length", fontsize=12, fontweight="bold")
plt.xticks(x, seq_lengths)
plt.yticks(np.arange(0, 141, 20))
plt.ylim(0, 145)
plt.grid(axis="y", linestyle="-", alpha=0.3)

# 5. 범례 설정 (2열로 배치) [cite: 42, 43, 44]
plt.legend(ncol=2, loc="upper left", fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig("memory_characterisation.svg", dpi=300, bbox_inches="tight")
