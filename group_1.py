import math
import random
import time
from utils import BST, menu, benchmark, print_table

SIZES = [100, 1000, 10000]
SAMPLES = 5

def ex01():
    tree = BST()
    tree.input_values()
    print(f"Árvore:\n{tree}")
    print(f"Em ordem: {tree.inorder()}")

def ex02():
    tree = BST()
    tree.input_values()
    print(f"Árvore:\n{tree}")
    target = int(input("Digite o valor a ser buscado: "))
    found = tree.search(target)
    print(f"Valor {target} {'encontrado' if found else 'não encontrado'} na árvore.")

def ex03():
    tree = BST()
    tree.input_values()
    print(f"Antes:\n{tree}")
    target = int(input("Digite o valor a ser removido: "))
    tree.remove(target)
    print(f"Depois:\n{tree}")

def ex04():
    tree = BST()
    tree.input_values()
    print(f"Árvore:\n{tree}")
    print(f"Altura: {tree.height()}")

def ex05():
    def random_values(n):
        return  random.sample(range(n * 10), n)

    def run_time(n):
        values = random_values(n)
        tree = BST()
        start = time.perf_counter()
        for v in values:
            tree.insert(v)
        t_insert = time.perf_counter() - start
        start = time.perf_counter()
        for v in values:
            tree.search(v)
        t_search = time.perf_counter() - start
        return t_insert, t_search

    def run_insert(n):
        values = random_values(n)
        tree = BST()
        for v in values:
            tree.insert(v)

        return (
            tree.comparisons,
            tree.writes,
            tree.steps,
            tree.steps / (n * math.log2(n)),
            tree.height()
        )

    def run_search(n):
        values = random_values(n)
        tree = BST()
        for v in values:
            tree.insert(v)
        tree.reset_counters()
        for v in values:
            tree.search(v)

        return (
            tree.comparisons,
            tree.writes,
            tree.steps,
            tree.steps / (n * math.log2(n))
        )

    metrics = [
        "comparações",
        "inserções",
        "passos",
        "/ N * log2(N)"
    ]
    insert_metrics = metrics + ["altura"]
    fmt = [".0f"] * 4 + [".2f"]
    insert_fmt = fmt + [".1f"]

    time_fmt = [".0f"] + [".5f"] * 4

    rows, headers = benchmark(
            axes={"N": SIZES},
            measure_fn=run_time,
            metrics= ["inserção (s)", "busca (s)"],
            samples=SAMPLES,
            dp=True
        )
    print_table(
        title=f"BST: Tempo de N operações ({SAMPLES} amostras, média e desvio padrão)",
        rows=rows,
        headers=headers,
        floatfmt=time_fmt
    )

    rows, headers = benchmark(
        axes={"N": SIZES},
        measure_fn=run_insert,
        metrics= insert_metrics,
        samples=SAMPLES
    )
    print_table(
        title=f"BST: Inserção de N valores aleatórios (média de {SAMPLES} amostras)",
        rows=rows,
        headers=headers,
        floatfmt=insert_fmt
    )

    rows, headers = benchmark(
        axes={"N": SIZES},
        measure_fn=run_search,
        metrics= metrics,
        samples=SAMPLES
    )
    print_table(
        title=f"BST: Busca de N valores aleatórios (média de {SAMPLES} amostras)",
        rows=rows,
        headers=headers,
        floatfmt=fmt
    )

def main():
    menu(
        title="Grupo 1 - BST",
        options={
            "Ex01 - Inserir N valores": ex01,
            "Ex02 - Buscar valor": ex02,
            "Ex03 - Remover valor": ex03,
            "Ex04 - Altura da árvore": ex04,
            "Ex05 - Tempos de busca e inserção para N = 100, 1000 e 10000": ex05,
        }
    )

if __name__ == "__main__":
    main()
