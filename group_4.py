from utils import menu, Trie

PREFIXES = [
    "10.0.0.0/8",
    "192.168.0.0/16",
    "192.168.1.0/24"
]

DEST_IPS = [
    "192.168.1.10",
    "192.168.5.5",
    "10.20.30.40",
    "8.8.8.8"
]

ROUTES = [
    ("0.0.0.0/0", "eth0 (rota padrão)"),
    ("10.0.0.0/8", "eth1"),
    ("192.168.0.0/16", "eth2"),
    ("192.168.1.0/24", "eth3")
]

def ip_to_int(ip):
    a, b, c , d = ip.split(".")
    return int(a) * 256**3 + int(b) * 256**2 + int(c) * 256 + int(d)

def int_to_ip(x):
    a = x // 256**3
    b = x // 256**2 % 256
    c = x // 256 % 256
    d = x % 256

    return f"{a}.{b}.{c}.{d}"

def network(x, size):
    block = 2**(32 - size)
    return x // block * block

def parse_prefix(prefix):
    ip, size = prefix.split("/")
    size = int(size)
    return network(ip_to_int(ip), size), size

def insert_prefix(table, prefix):
    net, size = parse_prefix(prefix)
    if size not in table:
        table[size] = set()
    table[size].add(net)

def ip_bits(ip):
    return f"{ip_to_int(ip):032b}"

def prefix_bits(prefix):
    net, size = parse_prefix(prefix)
    return f"{net:032b}"[:size]

def build_table():
    trie = Trie()
    for prefix, interface in ROUTES:
        trie.insert(prefix_bits(prefix), (prefix, interface))
    return trie

def ex01():
    table = {}
    for p in PREFIXES:
        insert_prefix(table, p)
    insert_prefix(table, "192.168.1.77/24")
    for size in sorted(table):
        for net in table[size]:
            print(f"{int_to_ip(net)}/{size}")

def ex02():
    trie = Trie()
    for p in PREFIXES:
        trie.insert(prefix_bits(p), p)
    not_found_prefixes = [
        "192.168.2.0/24",
        "10.0.0.0/16",
        "192.168.0.0/20"
    ]
    for p in PREFIXES + not_found_prefixes:
        print(f"{p} -> {trie.search(prefix_bits(p))}")

def ex03():
    trie = Trie()
    for p in PREFIXES:
        trie.insert(prefix_bits(p), p)
    for ip in DEST_IPS:
        found = trie.all_prefixes(ip_bits(ip))
        if found:
            best = found[-1]
        else:
            best = None
        print(f"{ip} compatíveis: {found} LPM: {best}")

def ex04():
    trie_values = build_table().values()
    for prefix, interface in trie_values:
        print(f"{prefix} -> {interface}")

def ex05():
    trie = build_table()
    for ip in DEST_IPS:
        route = trie.longest_prefix(ip_bits(ip))
        if route is None:
            print(f"{ip} -> sem rota")
        else:
            prefix, interface = route
            print(f"{ip} -> {prefix} saída: {interface}")

def main():
    menu(
        title="Grupo 4 - Prefixos IPv4 e Trie",
        options={
            "Ex01 - Inserir prefixos (dicionário)": ex01,
            "Ex02 - Trie binária + busca": ex02,
            "Ex03 - LPM (todos os compatíveis mais o maior)": ex03,
            "Ex04 - Trie com tabela de roteamento": ex04,
            "Ex05 - LPM na Trie": ex05
        }
    )

if __name__ == "__main__":
    main()
