import ipaddress
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

def to_bits(ip):
    return bin(int(ip))[2:].zfill(32)

def ip_bits(ip):
    return to_bits(ipaddress.ip_address(ip))

def prefix_bits(prefix):
    network = ipaddress.ip_network(prefix, strict=False)
    return to_bits(network.network_address)[:network.prefixlen]

def network_name(prefix):
    return str(ipaddress.ip_network(prefix, strict=False))

def build_trie(prefixes):
    trie = Trie()
    for p in prefixes:
        trie.insert(prefix_bits(p), network_name(p))
    return trie

def build_table():
    trie = Trie()
    for prefix, interface in ROUTES:
        trie.insert(prefix_bits(prefix), (prefix, interface))
    return trie

def ex01():
    trie = build_trie(PREFIXES + ["192.168.1.77/24"])
    for p in trie.values():
        print(p)

def ex02():
    trie = build_trie(PREFIXES)
    not_found_prefixes = [
        "192.168.2.0/24",
        "10.0.0.0/16",
        "192.168.0.0/20"
    ]
    for p in PREFIXES + not_found_prefixes:
        print(f"{p} -> {trie.search(prefix_bits(p))}")

def ex03():
    trie = build_trie(PREFIXES)
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
            "Ex01 - Inserir prefixos na Trie": ex01,
            "Ex02 - Trie binária + busca": ex02,
            "Ex03 - LPM (todos os compatíveis mais o maior)": ex03,
            "Ex04 - Trie com tabela de roteamento": ex04,
            "Ex05 - LPM na Trie": ex05
        }
    )

if __name__ == "__main__":
    main()
