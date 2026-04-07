# Mamba SSM Benchmarks

Hardware: Ada server — Intel Xeon (48 threads) + NVIDIA RTX 6000 Ada (48GB), CUDA 13.2, Driver 595.

## GPU Inference (T=1 step, default config: d_model=128, 3 layers, 366K params)

| Batch | No Graph | CUDA Graph |
|-------|----------|------------|
| B=1   | 124 us   | **79 us**  |
| B=4   | 147 us   | 100 us     |
| B=16  | 149 us   | 104 us     |
| B=64  | 157 us   | 116 us     |
| B=128 | 177 us   | 140 us     |

CUDA Graph eliminates kernel launch overhead (~45 us saved per step).

## GPU Training (default config, B=1, T=32)

| | Time |
|---|---|
| Forward | 589 us |
| Forward + Backward | 1,640 us |

## CPU Inference (T=1 step, B=1)

| Config | d_model | layers | params | us/step |
|--------|---------|--------|--------|---------|
| small  | 64      | 2      | 70K    | 24.9    |
| default| 128     | 3      | 366K   | 83.6    |
| medium | 256     | 4      | 1.8M   | 361     |
| large  | 512     | 6      | 10.4M  | 2,285   |

## CPU Training (B=1, T=32)

| Config | d_model | layers | Forward | Backward | Total |
|--------|---------|--------|---------|----------|-------|
| small  | 64      | 2      | 1,195 us | 1,964 us | 3,160 us |
| default| 128     | 3      | 4,074 us | 11,801 us | 15,874 us |
| medium | 256     | 4      | 13,345 us | 54,683 us | 68,028 us |
| large  | 512     | 6      | 67,508 us | 571,632 us | 639,140 us |

## CPU Parallel Training (default config, T=32, 48 threads)

| Batch | Forward | Backward | Total | Samples/sec |
|-------|---------|----------|-------|-------------|
| B=16  | 10,524 us | 27,509 us | 38,033 us | 421 |
| B=64  | 21,021 us | 66,050 us | 87,071 us | 735 |
| B=128 | 35,593 us | 103,177 us | 138,770 us | 922 |

## Speedups

| | GPU vs CPU |
|---|---|
| Inference B=1 (CUDA Graph) | **1.1x** |
| Training Fwd+Bwd T=32 | **9.8x** |

Zero heap allocations per inference step. All buffers pre-allocated.
