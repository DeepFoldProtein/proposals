# Python Scripts Data Summary

This document contains only the data definitions and relevant descriptions extracted from the Python scripts in `/Users/vv137/works/optimized`.

## Table of Contents
- [activation_memory.py](#activation_memorypy)
- [kernel_launch.py](#kernel_launchpy)
- [memory.py](#memorypy)
- [memory_characterisation.py](#memory_characterisationpy)
- [mgnify.py](#mgnifypy)
- [speed.py](#speedpy)
- [train.py](#trainpy)

---

## activation_memory.py
```python
# 1. 데이터 설정 [cite: 114, 116-119]
seq_lengths = ["64", "96", "128", "160", "192"]

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
```

---

## kernel_launch.py
```python
# 1. 데이터 설정 (BLOOM 제외, AF3만 포함) [cite: 114-119]
labels = [
    "aten::linear",
    "aten::layer_norm",
    "aten::native_layer_norm",
    "aten::sigmoid",
    "aten::silu",
]

# AF3 호출 횟수 데이터 [cite: 100-103, 106]
# GELUFunction은 AF3 데이터가 없으므로 0으로 설정
af3_counts = [37538, 13225, 12686, 10903, 2601]
```

---

## memory.py
```python
# 1. 데이터 설정 (시각적 근사치 반영) [cite: 34-39]
labels = ["128", "256", "384", "512", "640", "768"]

# 모델별 메모리 사용량 (GB) [cite: 22-23, 25-30]
# 0으로 표시된 부분은 OOM(Out of Memory) 지점입니다.
y_eager = [22, 40, 61, 95, 0, 0]
y_inductor = [22, 40, 61, 95, 0, 0]
y_optimized = [22, 37, 52, 78, 109, 139]

# H200 Limit
limit_val = 141.4  # 일반적인 H200 메모리 한계치
```

---

## memory_characterisation.py
```python
# 1. 데이터 설정 [cite: 63-67]
seq_lengths = ["64", "96", "128", "160", "192"]

# 각 구성 요소별 메모리 사용량 (GB) [cite: 46, 53, 56, 58, 60, 61]
activation_memory = [17.8, 33.5, 52.7, 77.2, 107.4]
model_weights = [3.1] * 5  # 추정치 (고정)
gradients = [3.1] * 5  # 추정치 (고정)
optimizer_states = [4.08] * 5  # 이미지에 명시된 값
```

---

## mgnify.py
```python
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

# ----------------------------
# FLOPS scaling (very rough)
# ----------------------------
# Peak BF16/FP16 Tensor Core FLOPS (dense) used for scaling
A100_TFLOPS = 312.0
H200_TFLOPS = 1979.0
```

---

## speed.py
```python
# 1. 데이터 설정 (시각적 근사치) [cite: 14-20]
x = [128, 256, 384, 512, 640, 768]

# 각 모델별 실행 시간 (s) 추정 데이터 [cite: 1, 4, 6]
y_eager = [6.6, 8.1, 10.5, 14.7, None, None]  # PyTorch (Eager Mode)
y_inductor = [6.7, 8.2, 10.9, 15.0, None, None]  # PyTorch (Inductor)
# y_optimized = [3.7, 4.6, 7.3, 8.5, 13.2, 17.8]  # MegaFold
y_optimized = [3.7, 4.6, 6.8, 8.5, 13.2, 17.8]  # MegaFold
```

---

## train.py
```python
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
```
