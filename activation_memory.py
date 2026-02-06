import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Monospace"

# 1. 데이터 설정 [cite: 114, 116-119]
seq_lengths = ["64", "96", "128", "160", "192"]
x = np.arange(len(seq_lengths))
width = 0.6

# 각 연산자별 메모리 점유량 (GB) [cite: 99, 102-105, 107-108, 110-112]
# 'Other Operators'는 전체 합계에서 명시된 두 값을 뺀 잔여값으로 계산함
transition_op = [4.93, 8.56, 12.9, 18.4, 24.4]
evo_attention_op = [5.96, 11.7, 19.4, 30.3, 44.5]
other_ops = [
    6.91,
    13.24,
    20.4,
    28.5,
    38.5,
]  # 합계(17.8, 33.5, 52.7, 77.2, 107.4) 기준 역산

# 2. 그래프 그리기 (누적 막대 그래프)
plt.figure(figsize=(10, 6))

# 아래에서부터 위로 쌓기
p1 = plt.bar(
    x, other_ops, width, label="Other Operators", color="#0066FF", edgecolor="black"
)
p2 = plt.bar(
    x,
    evo_attention_op,
    width,
    bottom=other_ops,
    label="EvoAttention Operator",
    color="#33CC66",
    edgecolor="black",
)
p3 = plt.bar(
    x,
    transition_op,
    width,
    bottom=np.array(other_ops) + np.array(evo_attention_op),
    label="Transition Operator",
    color="#FF0066",
    edgecolor="black",
)

# 3. 수치 라벨 추가 (이미지에 표시된 값들 배치) [cite: 99, 102-105, 107-108, 110-112]
for i in range(len(x)):
    break
    # Transition Operator 값 (상단 막대 오른쪽)
    total_height = other_ops[i] + evo_attention_op[i] + transition_op[i]
    plt.text(
        i + 0.35,
        total_height - (transition_op[i] / 2),
        f"{transition_op[i]}",
        va="center",
        fontsize=11,
    )

    # EvoAttention Operator 값 (중간 막대 오른쪽)
    mid_height = other_ops[i] + evo_attention_op[i]
    plt.text(
        i + 0.35,
        mid_height - (evo_attention_op[i] / 2),
        f"{evo_attention_op[i]}",
        va="center",
        fontsize=11,
    )

# 4. 스타일 및 레이블 설정
plt.ylabel("Activation Memory (GB)", fontsize=12, fontweight="bold")
plt.xlabel("Sequence Length", fontsize=12, fontweight="bold")
plt.xticks(x, seq_lengths)
plt.yticks(np.arange(0, 141, 20))
plt.ylim(0, 145)
plt.grid(axis="y", linestyle="-", alpha=0.3)

# 5. 범례 설정
plt.legend(loc="upper left", fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig("activation_memory.svg", dpi=300, bbox_inches="tight")
