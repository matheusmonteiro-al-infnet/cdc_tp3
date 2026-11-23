import multiprocessing
import os
import random
from numba import njit, prange, get_num_threads, threading_layer
from utils import menu, measure, benchmark, print_table, Node

TREE_SIZE = 1000000
PRIME_LIMITS = [100000, 1000000, 5000000]
SAMPLES = 5
subtrees = []
MAX_PROCESSES = os.cpu_count()
PROCESSES = []
p = 1

while p <= MAX_PROCESSES:
    PROCESSES.append(p)
    p *= 2

def insert_random(root, value):
    current = root
    while True:
        if random.random() < 0.5:
            if current.left is None:
                current.left = Node(value)
                return
            current = current.left
        else:
            if current.right is None:
                current.right = Node(value)
                return
            current = current.right

def build_tree(n):
    values = random.sample(range(n * 10), n)
    root = Node(values[0])
    for v in values[1:]:
        insert_random(root, v)
    return root, values

def subtree_max(node):
    best = -1
    stack = []
    if node is not None:
        stack.append(node)
    while stack:
        current = stack.pop()
        if current.value > best:
            best = current.value
        if current.left is not None:
            stack.append(current.left)
        if current.right is not None:
            stack.append(current.right)
    return best

def split_tree(root):
    top = [root]
    subtrees = []
    for child in (root.left, root.right):
        if child is not None:
            top.append(child)
            subtrees.append(child.left)
            subtrees.append(child.right)
    return top, subtrees

def worker(t):
    return subtree_max(subtrees[t])

def tree_max_par(pool, top):
    results = pool.map(worker, range(len(subtrees)))
    best = max(results)
    for node in top:
        if node.value > best:
            best = node.value
    return best

def ex01():
    root, values = build_tree(TREE_SIZE)
    m = subtree_max(root)
    print(f"Maior: {m} correto: {m == max(values)}")

    def run(n):
        return measure(subtree_max, root) * 1000,
    rows, headers = benchmark(
        axes={"N": [TREE_SIZE]},
        measure_fn=run,
        metrics=["seq (ms)"],
        samples=SAMPLES,
        dp=True
    )
    print_table(
        title=f"Maior valor, sequencial ({SAMPLES} amostras, média e dp)",
        rows=rows,
        headers=headers
    )

def ex04():
    global subtrees
    root, values = build_tree(TREE_SIZE)
    top, subtrees = split_tree(root)
    print(f"Núcleos: {MAX_PROCESSES} subárvores: {len(subtrees)}")
    ctx = multiprocessing.get_context("fork")
    pools = {}
    for p in PROCESSES:
        pools[p] = ctx.Pool(p)

    m_seq = subtree_max(root)
    m_par = tree_max_par(pools[PROCESSES[-1]], top)
    print(f"Maior seq = {m_seq} maior par = {m_par} ok: {m_seq == m_par}")

    def run(processes):
        t_seq = measure(subtree_max, root, runs=3)
        t_par = measure(tree_max_par, pools[processes], top, runs=3)
        speedup = t_seq / t_par
        return (
            t_seq * 1000,
            t_par * 1000,
            speedup,
            speedup / processes * 100
        )
    for p in PROCESSES:
        run(p)
    rows, headers = benchmark(
        axes={"Processos": PROCESSES},
        measure_fn=run,
        metrics=["seq (ms)", "par (ms)", "speedup", "eficiência (%)"],
        samples=SAMPLES,
        dp=True
    )
    for p in pools.values():
        p.close()
    print_table(
        title=f"Maior valor, N={TREE_SIZE}, seq x par ({SAMPLES} amostras, média e dp)",
        rows=rows,
        headers=headers
    )

@njit
def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True

@njit
def count_primes_seq(n):
    c = 0
    for x in range(2, n + 1):
        if is_prime(x):
            c += 1
    return c

@njit(parallel=True)
def count_primes_par(n):
    c = 0
    for x in prange(2, n + 1):
        if is_prime(x):
            c += 1
    return c

def ex02():
    count_primes_seq(10)
    for n in PRIME_LIMITS:
        print(f"N={n} primos={count_primes_seq(n)}")
    def run(n):
        return measure(count_primes_seq, n, runs=1) * 1000,
    rows, headers = benchmark(
        axes={"N": PRIME_LIMITS},
        measure_fn=run,
        metrics=["seq (ms)"],
        samples=SAMPLES,
        dp=True
    )
    print_table(
        title=f"Primos, sequencial ({SAMPLES} amostras, média e dp)",
        rows=rows,
        headers=headers
    )

def ex03():
    count_primes_seq(10)
    count_primes_par(10)
    print(f"Número de Threads: {get_num_threads()}")
    print(f"Motor de threads: {threading_layer()}")
    for n in PRIME_LIMITS:
        c_seq = count_primes_seq(n)
        c_par = count_primes_par(n)
        print(f"N={n} primos={c_par} ok={c_seq == c_par}")
    def run(n):
        t_seq = measure(count_primes_seq, n, runs=1)
        t_par = measure(count_primes_par, n, runs=1)
        return (
            t_seq * 1000,
            t_par * 1000,
            t_seq / t_par
        )
    rows, headers = benchmark(
        axes={"N": PRIME_LIMITS},
        measure_fn=run,
        metrics=["seq (ms)", "par (ms)", "speedup"],
        samples=SAMPLES,
        dp=True
    )
    print_table(
        title=f"Primos, seq x par ({SAMPLES} amostras, média e dp)",
        rows=rows,
        headers=headers
    )

def main():
    menu(
        title="Grupo 3 - Árvore e primos",
        options={
            "Ex01 - Maior valor da árvore (seq)": ex01,
            "Ex02 - Contar primos (seq)": ex02,
            "Ex03 - Contar primos (par)": ex03,
            "Ex04 - Maior valor da árvore (par)": ex04
        }
    )

if __name__ == "__main__":
    main()
