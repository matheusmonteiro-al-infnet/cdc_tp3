import os
import numpy as np
from numba import njit, prange
from utils import measure, benchmark, print_table, menu

MAX_THREADS = os.cpu_count()

SIZES = [500, 2000]
SAMPLES = 5

@njit
def invert_seq(img):
    rows, cols = img.shape
    out = np.empty_like(img)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = 255 - img[i, j]
    return out

@njit(parallel=True)
def invert_par(img):
    rows, cols = img.shape
    out = np.empty_like(img)
    for i in prange(rows):
        for j in range(cols):
            out[i, j] = 255 - img[i, j]
    return out

def generate_img(size):
    return np.random.randint(
        low=0,
        high=256,
        size=(size, size),
        dtype=np.uint8
    )

def ex01():
    warmup_img = generate_img(1000)
    invert_seq(warmup_img)
    invert_par(warmup_img)

    images = {}
    for size in SIZES:
        images[size] = generate_img(size)

    def run(size):
        t_seq = measure(invert_seq, images[size])
        t_par = measure(invert_par, images[size])
        speedup = t_seq / t_par
        return t_seq * 1e6, t_par * 1e6, speedup, speedup / MAX_THREADS * 100

    run(SIZES[0])

    rows, headers = benchmark(
        axes={"tamanhos": SIZES},
        measure_fn=run,
        metrics=["Tempo Seq. (us)", "Tempo Par. (us)", "Speedup", "Eficiência (%)"],
        samples=SAMPLES,
        dp=True
    )
    print_table(
        title=f"Inversão de pixels, {MAX_THREADS} threads (média de {SAMPLES} amostras)",
        rows=rows,
        headers=headers
    )

K = 100
N = 10000000

@njit
def freq_seq(v, k):
    count = np.zeros(k, dtype=np.int64)
    for x in v:
        count[x] += 1
    return count

@njit(parallel=True)
def freq_par(v, k, n_chunks):
    chunk = (len(v) + n_chunks - 1) // n_chunks
    partial = np.zeros((n_chunks, k), dtype=np.int64)
    for t in prange(n_chunks):
        start = t * chunk
        end = min(start + chunk, len(v))
        count = np.zeros(k, dtype=np.int64)
        for i in range(start, end):
            count[v[i]] += 1
        partial[t] = count
    return partial.sum(axis=0)

def ex02():
    v = np.random.randint(0, K, N)
    freq_seq(v[:10], K)
    freq_par(v[:10], K, MAX_THREADS)

    f_par = freq_par(v, K, MAX_THREADS)
    print("Primeiros valores:")
    for x in range(5):
        print(f"{x}: {f_par[x]}")

    def run(n):
        t_seq = measure(freq_seq, v, K)
        t_par = measure(freq_par, v, K, MAX_THREADS)
        speedup = t_seq / t_par
        return (
            t_seq * 1000,
            t_par * 1000,
            speedup,
            speedup / MAX_THREADS * 100
        )
    run(N)
    rows, headers = benchmark(
        axes={"N": [N]},
        measure_fn=run,
        metrics=["seq (ms)", "par (ms)", "speedup", "eficiência (%)"],
        samples=SAMPLES,
        dp=True
    )
    print_table(
        title=f"Frequência, K={K}, {MAX_THREADS} threads ({SAMPLES} amostras, média e dp)",
        rows=rows,
        headers=headers
    )

def main():
    menu(
        title="Grupo 2 - Numba",
        options={
            "Ex01 - Inversão de pixels (seq x par)": ex01,
            "Ex02 - Frequência de valores (paralelo)": ex02
        }
    )

if __name__ == "__main__":
    main()
