import time
from itertools import product
from tabulate import tabulate

def menu(title:str, options:dict):
    names = list(options)
    while True:
        print(f"\n{"="*5} {title} {"="*5}")
        for i, name in enumerate(names, start=1):
            print(f"{i} - {name}")
        print("0 - Voltar")
        choice = input("Opção: ")
        if choice == "0":
            return
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            func = options[names[int(choice) - 1]]
            func()
        else:
            print("Opção inválida")

def measure(func, *args, runs=10):
    start = time.perf_counter()
    for _ in range(runs):
        func(*args)
    return (time.perf_counter() - start) / runs

def average(values):
    return sum(values) / len(values)

def std_dev(values):
    a = average(values)
    total = 0
    for v in values:
        total += (v - a) ** 2
    return (total / len(values)) ** 0.5

def benchmark(axes, measure_fn, metrics, samples=1, dp=False):
    rows = []
    for combo in product(*axes.values()):
        results = []
        for _ in range(samples):
            results.append(measure_fn(*combo))
        row = list(combo)
        for i in range(len(metrics)):
            values = [r[i] for r in results]
            row.append(average(values))
            if dp:
                row.append(std_dev(values))
        rows.append(row)
    headers = list(axes)
    for name in metrics:
        headers.append(name)
        if dp:
            headers.append(name + " dp")
    return rows, headers

def print_table(title, rows, headers, floatfmt=".2f"):
    print(f"\n{title}")
    print(tabulate(
        rows,
        headers=headers,
        tablefmt="github",
        floatfmt=floatfmt)
    )

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
        self.__comparisons = 0
        self.__writes = 0

    def reset_counters(self):
        self.__comparisons = 0
        self.__writes = 0

    @property
    def comparisons(self):
        return self.__comparisons
    @property
    def writes(self):
        return self.__writes

    @property
    def steps(self):
        return self.__comparisons + self.__writes

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
            self.__writes += 1
            return
        current = self.root
        while True:
            self.__comparisons += 1
            if value < current.value:
                if current.left is None:
                    current.left = Node(value)
                    self.__writes += 1
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(value)
                    self.__writes += 1
                    return
                current = current.right

    def inorder(self):
        result = []
        stack = []
        current = self.root
        while current is not None or stack:
            while current is not None:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.value)
            current = current.right
        return result

    def search(self, value):
        current = self.root
        while current is not None:
            self.__comparisons += 1
            if value == current.value:
                return True
            self.__comparisons += 1
            if value < current.value:
                current = current.left
            else:
                current = current.right
        return False

    def remove(self, value):
        parent = None
        current = self.root
        while current is not None:
            self.__comparisons += 1
            if value == current.value:
                break
            parent = current
            self.__comparisons += 1
            if value < current.value:
                current = current.left
            else:
                current = current.right
        if current is None:
            return
        if current.left is not None and current.right is not None:
            succ_parent = current
            succ = current.right
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            current.value = succ.value
            self.__writes += 1
            current = succ
            parent = succ_parent
        if current.left is not None:
            child = current.left
        else:
            child = current.right
        if parent is None:
            self.root = child
        elif parent.left is current:
            parent.left = child
        else:
            parent.right = child
        self.__writes += 1
    def height(self):
        levels = -1
        level = []
        if self.root is not None:
            level.append(self.root)
        while level:
            levels += 1
            next_level = []
            for node in level:
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            level = next_level
        return levels

    def input_values(self):
        n = int(input("Quantos valores (N)? "))
        values = input(f"Digite os {n} valores separando por espaço: ").split()
        for i in range(n):
            self.insert(int(values[i]))

    def __str__(self):
        if self.root is None:
            return "Árvore vazia"
        values = self.inorder()
        width = max(len(str(v)) for v in values) + 1
        col = {}
        position = 0
        stack = []
        current = self.root
        while current is not None or stack:
            while current is not None:
                stack.append(current)
                current = current.left
            current = stack.pop()
            col[current] = position * width
            position += 1
            current = current.right
        lines = []
        level = [self.root]
        while level:
            value_line = ""
            branch_line = ""
            next_level = []
            for node in level:
                value_line += " " * (col[node] - len(value_line))
                value_line += str(node.value)
                if node.left is not None:
                    middle = (col[node.left] + col[node]) // 2
                    branch_line += " " * (middle - len(branch_line)) + "/"
                    next_level.append(node.left)
                if node.right is not None:
                    middle = (col[node] + col[node.right]) // 2
                    branch_line += " " * (middle - len(branch_line)) + "\\"
                    next_level.append(node.right)
            lines.append(value_line)
            if next_level:
                lines.append(branch_line)
            level = next_level
        return "\n".join(lines)

class TrieNode:
    def __init__(self):
        self.child = [None, None]
        self.value = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, bits, value):
        node = self.root
        for b in bits:
            i = int(b)
            if node.child[i] is None:
                node.child[i] = TrieNode()
            node = node.child[i]
        node.value = value

    def search(self, bits):
        node = self.root
        for b in bits:
            node = node.child[int(b)]
            if node is None:
                return False
        return node.value is not None

    def all_prefixes(self, bits):
        result = []
        node = self.root
        if node.value is not None:
            result.append(node.value)
        for b in bits:
            node = node.child[int(b)]
            if node is None:
                break
            if node.value is not None:
                result.append(node.value)
        return result

    def longest_prefix(self, bits):
        node = self.root
        best = node.value
        for b in bits:
            node = node.child[int(b)]
            if node is None:
                break
            if node.value is not None:
                best = node.value
        return best

    def values(self):
        result = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.value is not None:
                result.append(node.value)
            if node.child[1] is not None:
                stack.append(node.child[1])
            if node.child[0] is not None:
                stack.append(node.child[0])
        return result
