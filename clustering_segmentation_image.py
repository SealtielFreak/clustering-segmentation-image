import abc
import collections
import dataclasses
import itertools
import math
import typing

import numpy as np
import pandas as pd

from PIL import Image

T = typing.TypeVar("T", int, float)
Vector = typing.Tuple[T, T]
VectorComponent = typing.List[T]
Vertex = typing.List[typing.Tuple[T, T]]


def check_axis_intersect(that: Vector, z: T):
    x, y = that
    return y >= z >= x


def is_collision(that: Vertex, other: VectorComponent) -> bool:
    if len(that) != len(other):
        raise ValueError("Invalid arguments, different length of values")

    return all([check_axis_intersect(t, z) for t, z in zip(that, other)])


@dataclasses.dataclass
class VectorNode:
    data: VectorComponent
    value: T

    def __gt__(self, other):
        return self.value > other.value

    def __str__(self):
        return f"{self.data}, {self.value}"

    def __iter__(self):
        return iter([*self.data, self.value])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.vertex[item]

    def __setitem__(self, key, value):
        self.vertex[key] = value


def calc_distance(x: T, y: T):
    return math.fabs(y - x) / 2


def create_axis(_array: Vertex):
    all_data = []

    for arr in _array:
        d = calc_distance(*arr)

        all_data.append(((x, x + d), (x + d, y)))

    return all_data


def create_axis_node(arrays):
    return itertools.product(arrays)


@dataclasses.dataclass
class Node:
    vertex: Vertex
    nodes: list

    def append(self, node: VectorComponent):
        self.nodes.append(node)

    def is_collide(self, node: VectorComponent):
        return is_collision(self.vertex, node)

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def __hash__(self):
        _hash = hash(tuple(self.vertex))

        return _hash


@dataclasses.dataclass
class NodeValue:
    vertex: Vertex
    node: VectorNode

    def __gt__(self, other: VectorNode):
        return self.value > other.value

    @property
    def value(self):
        return self.node.value


class NodeContainerInterface(abc.ABC):
    @property
    @abc.abstractmethod
    def axis(self) -> Vertex: ...

    @property
    @abc.abstractmethod
    def children(self): ...

    @property
    @abc.abstractmethod
    def node(self) -> Node: ...

    @property
    @abc.abstractmethod
    def is_parent(self) -> bool: ...

    @abc.abstractmethod
    def insert(self, verx: VectorComponent): ...

    @abc.abstractmethod
    def sort(self): ...

    @abc.abstractmethod
    def __iter__(self): ...


class TreeNode:
    def __init__(self, axis: Vertex, limit_divisions=1):
        self.__children = {}
        self.__node = Node(vertex=axis, nodes=[])
        self.__axis = axis
        self.__limit_divisions = limit_divisions

    def __hash__(self):
        return hash(self.__node)

    @property
    def axis(self):
        return [*self.__axis]

    @property
    def children(self):
        return self.__children

    @property
    def node(self) -> Node:
        return self.__node

    @property
    def is_parent(self):
        return len(self.children) != 0

    def insert(self, verx: VectorComponent):
        return self._insert_recursive(verx)

    def sort(self):
        return self._get_iter_child_recursive()

    def __iter__(self):
        return iter(self._get_iter_child_recursive())

    def _insert_recursive(self, verx: VectorComponent):
        def create_vertex(verx):
            root_axis = collections.deque()

            for axis, c in zip(self.axis, verx):
                x, y = axis
                d = calc_distance(x, y)

                if c >= x and c <= (x + d):
                    root_axis.append((x, x + d))
                else:
                    root_axis.append((x + d, y))

            axis = list(root_axis)

            return axis

        tree = TreeNode(create_vertex(verx), self.__limit_divisions - 1)
        tree_key = hash(tree)

        if tree_key in self.__children:
            tree = self.__children[tree_key]
        else:
            self.__children[tree_key] = tree

        if self.__limit_divisions > 0:
            return tree.insert(verx)
        else:
            if not tree.node.is_collide(verx):
                raise ValueError(f"Vertex no collide: {tree.node} {verx}")

            tree.node.append(verx)

        return tree

    def _get_iter_child_recursive(self):
        def get_iter_child(root, nodes=None):
            if nodes is None:
                nodes = []

            for _, child in root.children.items():
                node = child.node

                if child.is_parent:
                    get_iter_child(child, nodes)
                else:
                    if len(child.node) > 0:
                        nodes.append(child.node)

            return nodes

        return get_iter_child(self, [])


def load_image(filename: str, mode: str = "RGB"):
    with Image.open(filename) as img:
        im = img.convert(mode)
        return im, np.asarray(im)


if __name__ == "__main__":
    img, img_arr = load_image("resources/b-source-0.bmp", "I")
    nodes = []

    for y, row in enumerate(img_arr):
        for x, p in enumerate(row):
            nodes.append(Node((x, y), p))

    mask_a, mask_b = [], []

    for n in nodes:
        if n.value != 0:
            mask_a.append(n)
        else:
            mask_b.append(n)
    
    print(mask_a)
