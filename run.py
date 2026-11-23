import group_1, group_2, group_3, group_4
from utils import menu

menu(
    title="Teste de Performance 3",
    options={
        "Grupo 1 - BST": group_1.main,
        "Grupo 2 - Numba (imagem e frequência)": group_2.main,
        "Grupo 3 - Árvore e primos (seq x par)": group_3.main,
        "Grupo 4 - Prefixos IPv4 e Trie": group_4.main,
    }
)