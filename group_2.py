import numpy as np
from numba import (
    njit,
    prange,
    get_num_threads,
    threading_layer,
    set_num_threads
)
from utils import measure, benchmark, print_table, menu

MAX_THREADS = get_num_threads()
THREADS = []
t = 1

while t <= MAX_THREADS:
    THREADS.append(t)
    t *= 2

SIZES = [500, 2000]
SAMPLES = 5

def invert(img):
    rows, cols = img.shape
    out = np.empty_like(img)
    for i in prange(rows):
        for j in range(cols):
            out[i, j] = 255 - img[i, j]
    return out

invert_seq = njit(invert)
invert_par = njit(parallel=True)(invert)

def generate_img(size):
    return np.random.randint(
        low=0,
        high=256,
        size=(size, size),
        dtype=np.uint8
    )

def ex01():
    print(f"Threads disponíveis: {get_num_threads()}")
    warmup_img = generate_img(1000)
    invert_seq(warmup_img)
    invert_par(warmup_img)
    print(f"Motor de Threads: {threading_layer()}")

    images = {}
    for size in SIZES:
        img = generate_img(size)
        ok = (invert_seq(img) == 255 - img).all()
        ok = ok and (invert_par(img) == 255 - img).all()
        print(f"{size}x{size} correto={ok}")
        images[size] = img

    def run(size, threads):
        set_num_threads(threads)
        t_seq = measure(invert_seq, images[size])
        t_par = measure(invert_par, images[size])
        speedup = t_seq / t_par
        return t_seq * 1e6, t_par * 1e6, speedup, speedup / threads * 100

    run(SIZES[0], MAX_THREADS)

    rows, headers = benchmark(
        axes={"tamanhos": SIZES, "threads": THREADS},
        measure_fn=run,
        metrics=["Tempo Seq. (us)", "Tempo Par. (us)", "Speedup", "Eficiência (%)"],
        samples=SAMPLES,
        dp=True
    )
    set_num_threads(MAX_THREADS)
    print_table(
        title=f"Inversão de pixels (média de {SAMPLES} amostras)",
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
def freq_par(v, k):
    n_threads = get_num_threads()
    chunk = (len(v) + n_threads - 1) // n_threads
    partial = np.zeros((n_threads, k), dtype=np.int64)
    for t in prange(n_threads):
        start = t * chunk
        end = min(start + chunk, len(v))
        count = np.zeros(k, dtype=np.int64)
        for i in range(start, end):
            count[v[i]] += 1
        partial[t] = count
    return partial.sum(axis=0)

def ex02():
    v = np.random.randint(0, K, N)
    print(f"Número de Threads: {get_num_threads()}")
    freq_seq(v[:10], K)
    freq_par(v[:10], K)
    print(f"Motor de threads: {threading_layer()}")

    f_seq = freq_seq(v, K)
    f_par = freq_par(v, K)
    print("Primeiros valores:")
    for x in range(5):
        print(f"{x}: {f_par[x]}")
    print(f"Iguais ao sequencial? {(f_seq == f_par).all()}")
    print(f"Soma das frequências = N? {f_par.sum() == N}")

    def run(n, threads):
        set_num_threads(threads)
        t_seq = measure(freq_seq, v, K)
        t_par = measure(freq_par, v, K)
        speedup = t_seq / t_par
        return (
            t_seq * 1000,
            t_par * 1000,
            speedup,
            speedup / threads * 100
        )
    run(N, MAX_THREADS)
    rows, headers = benchmark(
        axes={"N": [N], "threads": THREADS},
        measure_fn=run,
        metrics=["seq (ms)", "par (ms)", "speedup", "eficiência (%)"],
        samples=SAMPLES,
        dp=True
    )
    set_num_threads(MAX_THREADS)
    print_table(
        title=f"Frequência, K={K} ({SAMPLES} amostras, média e dp)",
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
