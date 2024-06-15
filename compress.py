from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *

# CITATIONS:
# 1.https://www.csfieldguide.org.nz/en/interactives/huffman-tree/
#   To better understand HuffmanTrees and visualize them and confirm that the
#    Huffman-trees being built are right.
# 2.https://www.youtube.com/watch?v=0kNXhFIEd_w
#   Video explaining on Huffman coding and Huffman trees to understand how
#   Huffman trees are made and why they are efficient.
# 3.https://cmps-people.ok.ubc.ca/ylucet/DS/Huffman.html
#   Huffman-tree visualizer giving realtime animation on how a huffman tree is
#   being built, useful for debugging and code analysis.


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    dict1 = {}
    for j in text:
        if j in dict1:
            dict1[j] += 1
        else:
            dict1[j] = 1
    return dict1


def _helper_build(node1: HuffmanTree, node2: HuffmanTree) -> \
        (HuffmanTree, HuffmanTree):
    """
    Helper function for build huffman trees in case where many
    symbols are of same frequency
    """
    if node1.symbol and node2.symbol is not None:
        return node1, node2
    elif node1.symbol is None and node2.symbol:
        return node2, node1
    elif node2.symbol is None and node1.symbol:
        return node1, node2
    else:
        return node2, node1


def _helper_build2(node1: HuffmanTree, node2: HuffmanTree) -> \
        (HuffmanTree, HuffmanTree):
    """
    Helper function to figure out left and right subtrees for each instance
    """
    left, right = node1, node2
    if node1.number > node2.number:
        left, right = node2, node1
    elif node1.number < node2.number:
        left, right = node1, node2
    elif node1.number == node2.number:
        if node1.symbol is None and node2.symbol is None:
            left, right = node1, node2
        else:
            left, right = _helper_build(node1, node2)
    return left, right


def _sorter(lst: list[HuffmanTree]) -> list:
    """
    Sorter function to sort priority queue often
    """
    i = 0
    while len(lst) > 1:
        if len(lst) == 2:
            break
        if i != len(lst) - 1:
            if lst[i].number > lst[i + 1].number:
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
            elif lst[i].number == lst[i + 1].number:
                lst[i], lst[i + 1] = _helper_build(lst[i], lst[i + 1])
            i += 1
        else:
            break
    return lst


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    sortlst = sorted(freq_dict.items(), key=lambda x: x[1])
    freq, p_q = dict(sortlst), []
    for symbol, freq1 in freq.items():
        temphuff, temphuff.number = HuffmanTree(symbol), freq1
        p_q.append(temphuff)
    if len(p_q) == 1:
        return HuffmanTree(None, left=p_q.pop())
    while len(p_q) >= 1:
        node1 = p_q.pop(0)
        if not p_q:
            return node1
        if len(p_q) == 1:
            node2 = p_q.pop()
        elif abs(node1.number - p_q[1].number) == \
                abs(node1.number - p_q[0].number):
            node2 = _helper_build(p_q[0], p_q[1])[0]
            p_q.remove(node2)
        elif abs(node1.number - p_q[1].number) < \
                abs(node1.number - p_q[0].number):
            node2 = p_q.pop(1)
        else:
            node2 = p_q.pop(0)
        left, right = _helper_build2(node1, node2)
        mnode = HuffmanTree(None, left, right)
        mnode.number = left.number + right.number
        p_q.insert(1, mnode)
        p_q = _sorter(p_q)
    return p_q[0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    tempd = {}
    _get_codeshelper(tree, "", tempd)
    tempd = dict(sorted(tempd.items()))
    return tempd


def _get_codeshelper(tree: HuffmanTree, code: str, codes: dict) -> None:
    """
    Helper function for get codes function
    """
    if tree is not None:
        if tree.symbol is not None:
            codes[tree.symbol] = code
        else:
            _get_codeshelper(tree.left, code + "0", codes)
            _get_codeshelper(tree.right, code + "1", codes)


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    cnt = 0
    lst = _postorderhelper(tree)
    for i in lst:
        i.number = cnt
        cnt += 1


def _postorderhelper(tree: HuffmanTree) -> list:
    """
    Helper function for returning internal nodes of tree in post order
    """
    if tree is None:
        return []
    elif tree.left and tree.right is None:
        if not tree.is_leaf():
            return [tree.symbol]
    else:
        if tree.is_leaf():
            ret = []
        else:
            ret = [tree]
        return _postorderhelper(tree.left) + _postorderhelper(tree.right) + \
            ret
    return []


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    dct = get_codes(tree)
    tot, cnt = 0, 0
    for i in freq_dict:
        cnt += freq_dict[i]
        tot += len(dct[i]) * freq_dict[i]
    return tot / cnt


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    temp = ""
    lst, res = [], []
    for a in text:
        for j in codes[a]:
            temp += j
            if len(temp) == 8:
                lst.append(temp)
                temp = ""
    lst.append(temp)
    res = [(bits_to_byte(i)) for i in lst]
    return bytes(res)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    lst, fin = _postorderhelper(tree), []
    for tr in lst:
        if tr.left.is_leaf():
            fin.append(0)
            fin.append(tr.left.symbol)
        elif not tr.left.is_leaf():
            fin.append(1)
            fin.append(tr.left.number)
        if tr.right.is_leaf():
            fin.append(0)
            fin.append(tr.right.symbol)
        elif not tr.right.is_leaf():
            fin.append(1)
            fin.append(tr.right.number)
    return bytes(fin)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))

    """
    temp_queue = []
    for tr in node_lst:
        if tr.l_type == 0 and tr.r_type == 0:
            left, right = HuffmanTree(tr.l_data), HuffmanTree(tr.r_data)
        elif tr.r_type == 1 and tr.l_type == 0:
            left, right = HuffmanTree(tr.l_data), HuffmanTree(None)
            right.number = tr.r_data
        elif tr.r_type == 0 and tr.l_type == 1:
            left, right = HuffmanTree(None), HuffmanTree(tr.r_data)
            left.number = tr.l_data
        else:
            left, right = HuffmanTree(None), HuffmanTree(None)
            left.number, right.number = tr.l_data, tr.r_data
        temp_queue.append(HuffmanTree(None, left, right))
    for item in temp_queue:
        if item.left.number is not None:
            item.left = temp_queue[item.left.number]
        if item.right.number is not None:
            item.right = temp_queue[item.right.number]
    return temp_queue[root_index]


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    root, temp_queue = node_lst[root_index], []
    for tr in node_lst:
        if tr.l_type == 0 and tr.r_type == 0:
            item = HuffmanTree(None, HuffmanTree(tr.l_data),
                               HuffmanTree(tr.r_data))
        elif tr.l_type == 0 and tr.r_type != 0:
            item = HuffmanTree(None, HuffmanTree(tr.l_data), temp_queue.pop(0))
        elif tr.l_type != 0 and tr.r_type == 0:
            item = HuffmanTree(None, temp_queue.pop(0), HuffmanTree(tr.r_data))
        else:
            item = HuffmanTree(None, temp_queue.pop(), temp_queue.pop())
        temp_queue.insert(0, item)
    if isinstance(root.l_data, int):
        return temp_queue[-1]
    else:
        return temp_queue[-1]


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    dct = dict((a, b) for b, a in get_codes(tree).items())
    temp, lst = "", []
    ljk = "".join([byte_to_bits(st) for st in text])
    for i in ljk:
        temp += i
        if temp in dct:
            lst.append(dct[temp])
            temp = ""
    return bytes(lst[:size])


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    sortlst = sorted(freq_dict.items(), key=lambda x: x[1])
    lst2 = list(dict(sortlst))
    lst, temp = _levelorder(tree), []
    for ls in lst:
        for i in ls:
            if i.symbol is not None:
                temp.insert(0, i)
    for item in temp:
        if item.symbol != lst2[0]:
            item.symbol = lst2[0]
        lst2.pop(0)


def _levelorder(tree: HuffmanTree) -> list:
    """
    Helper function to return Huffman tree in level order
    """
    if tree is None:
        return []
    temp = [tree]
    fin = []
    while temp:
        lvl = []
        for i in temp:
            c_node = temp.pop(0)
            lvl.append(c_node)
            if c_node.left and i is not None:
                temp.append(c_node.left)
            if c_node.right is not None:
                temp.append(c_node.right)
        fin.append(lvl)
    return fin


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
