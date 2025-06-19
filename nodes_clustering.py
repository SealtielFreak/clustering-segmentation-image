import typing
import collections
import itertools
import math
import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from pandas import DataFrame

T = typing.TypeVar("T", int, float)


class Point(typing.Generic[T]):
    def __init__(self, position: tuple[T, T]):
        self.x, self.y = position

    @property
    def position(self):
        return self.x, self.y

    def __iter__(self) -> typing.Iterator[T]:
        return iter((self.x, self.y))

    def __getitem__(self, item):
        data = [self.x, self.y]
        return data[item]

    def __setitem__(self, key, value):
        data = [self.x, self.y]
        data[key] = value

    def __str__(self):
        return str((self.x, self.y))


class Circle(Point[T]):
    def __init__(self, position: tuple[T, T], radius: T = 0) -> None:
        super().__init__(position)
        self.radius = radius


class Node(Point[T]):
    def __init__(self, position: tuple[T, T], label: str) -> None:
        super().__init__(position)
        self.children: typing.List[Node] = []
        self.__label = label

    @property
    def label(self) -> str:
        return self.__label

    def __hash__(self):
        return hash(self.label)

    def clear(self):
        self.children.clear()


class BrushstrokeNode(Circle[T]):
    def __init__(self, img_source: np.ndarray, position: tuple[T, T], radius: T, sub_matrix_size=1):
        super().__init__(position, radius)

        self.__img_source = img_source
        self.fill_color = (0, 0, 0, 0)
        self.sub_matrix_size = sub_matrix_size

        self.update()

    def update(self):
        (x, y), r = self.position, self.radius
        sub_m = self.__img_source[y:y + self.sub_matrix_size, x: x + self.sub_matrix_size]
        media_color = np.median(sub_m, axis=0).astype(int)[0]

        self.fill_color = (*media_color, self.alpha)

    @property
    def alpha(self):
        return self.fill_color[3]

    @property
    def rgb(self) -> tuple[int, int, int]:
        return self.fill_color[0:3]

    @property
    def xyr(self):
        (x, y), r = self.position, self.radius
        return x - r, y - r, x + r, y + r

    def draw(self, canvas):
        canvas.ellipse(self.xyr, fill=self.rgb)

        return canvas


class Rect(typing.Generic[T]):
    def __init__(self, position: typing.Tuple[T, T], size: typing.Tuple[T, T]):
        self.position = position
        self.size = size

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def right(self):
        return self.position[0] + self.width

    @property
    def left(self):
        return self.position[0]

    @property
    def top(self):
        return self.position[1]

    @property
    def bottom(self):
        return self.position[1] + self.height

    def __str__(self):
        x, y = self.position

        return f"[{x}, {y}, {self.right}, {self.bottom}]"

    def __iter__(self):
        x, y = self.position

        return iter((x, y, self.right, self.bottom))


def collision_point(a: Rect[T], x: T, y: T):
    x_axis_left = x > a.left
    x_axis_right = x < a.right
    x_axis = x_axis_left and x_axis_right

    y_axis_top = y > a.top
    y_axis_bottom = y < a.bottom
    y_axis = y_axis_top and y_axis_bottom

    return x_axis and y_axis


def collision_circle(a: Circle, b: Circle):
    x1, y1 = a.position
    x2, y2 = b.position

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance <= a.radius + b.radius


def collision_rect(a: Rect[T], b: Rect[T]):
    return (a.bottom > b.top or b.top < a.bottom) and (a.right > b.left or a.left < b.right)


def is_collision(that: Rect[T], other: Rect[T] | Point) -> bool:
    collision_aabb: typing.Callable[[Rect[T], Rect[T]], bool] = lambda a, b: (
            (a.bottom > b.top or b.top < a.bottom) and
            (a.right > b.left or a.left < b.right)
    )

    if isinstance(other, Rect):
        return collision_rect(that, other)

    _x, _y, = other

    return collision_point(that, _x, _y)


def create_child(parent: Rect[T], size_limit: typing.Tuple[T, T]):
    w, h = parent.width / 2, parent.height / 2
    x, y = parent.position

    child = []

    if w >= size_limit[0] and h >= size_limit[1]:
        size = w, h

        rects = [
            Rect((x, y), size),
            Rect((x + w, y), size),
            Rect((x, y + h), size),
            Rect((x + w, y + h), size),
        ]

        for rect in rects:
            child.append(
                TreeNode(rect, size_limit)
            )

    return child


class TreeNode(typing.Generic[T]):
    def __init__(self, rect: Rect[T], size_limit=None):
        self.__rect = rect
        self.__points = collections.deque()

        size_limit_child = size_limit if size_limit else (
            rect.size[0] / 2, rect.size[1] / 2
        )

        self.__child = create_child(
            self.__rect,
            size_limit_child
        )

    @property
    def rect(self):
        return self.__rect

    @property
    def child(self):
        return self.__child

    @property
    def points(self):
        return self.__points

    def insert_point(self, point: Point):
        if not len(self.child) > 0:
            self.points.append(point)
        else:
            for c in self.child:
                if is_collision(c.rect, point):
                    return c.insert_point(point)

    def __str__(self):
        return str(self.rect)

    def __iter__(self):
        def iter_child(child: list[TreeNode], points=None):
            if points is None:
                points = []

            for c in child:
                if len(c.child) != 0:
                    iter_child(c.child, points)
                else:
                    if len(c.points) > 0:
                        points.append(list(c.points))

            return points

        return iter(iter_child(self.child))

    def get_points_with_root(self):
        def iter_child(child: list[TreeNode], points=None):
            if points is None:
                points = []

            for c in child:
                if len(c.child) != 0:
                    iter_child(c.child, points)
                else:
                    if len(c.points) > 0:
                        points.append((c, list(c.points)))

            return points

        return list(iter_child(self.child))

    def sort(self, root=False):
        if root:
            return self.get_points_with_root()

        return list(self)


def generate_random_nodes(n):
    nodes = []
    n_chr = 0

    for i, c in enumerate(range(n)):
        x, y = np.random.uniform(0, 1, size=2)

        if i > len(string.ascii_uppercase) - 1:
            n_chr += 1

        if n_chr > len(string.ascii_uppercase) - 1:
            n_chr = 0

        nodes.append(Node((x, y), f"{string.ascii_uppercase[n_chr]}{c}"))

    return nodes


DEFAULT_NODES_LIST = [
    Node((0, 0), 'A'),
    Node((1, 3), 'B'),
    Node((2, 2), 'C'),
    Node((8, 6), 'D'),
    Node((7, 7), 'E'),
    Node((7, 3), 'F'),
]

RANDOM_NODES_LIST = generate_random_nodes(25)

if __name__ == "__main__":
    all_nodes = RANDOM_NODES_LIST

    all_nodes_position_x = [n[0] for n in all_nodes]
    all_nodes_position_y = [n[1] for n in all_nodes]
    all_nodes_position_labels = [n.label for n in all_nodes]

    all_nodes_dist = []

    for a_it, p_a in enumerate(all_nodes):
        for b_it, p_b in enumerate(all_nodes[a_it + 1:]):
            all_nodes_dist.append((
                math.dist(p_a.position, p_b.position), p_a, p_b
            ))

    all_nodes_dist = sorted(all_nodes_dist, key=lambda x: x[0])
    max_combinations = (list(itertools.combinations(all_nodes_position_labels, 2)))

    dataframe_all_routes = DataFrame([(dist, a.label, b.label) for dist, a, b in all_nodes_dist])
    dataframe_optimized_routes = DataFrame([(dist, a.label, b.label) for dist, a, b in all_nodes_dist[:len(all_nodes)]])

    print(dataframe_all_routes)
    print(dataframe_optimized_routes)


    max_distance = all_nodes_dist[-1][0] / 5
    print(f"Max distance: {max_distance}")

    for dist, a, b in all_nodes_dist:
        x0, y0 = a.position
        x1, y1 = b.position

        dx = (x1 - x0)
        dy = (y1 - y0)

        if dist > max_distance:
            continue

        dist_c = 1 if dist <= 1 else 1 / dist
        color = 1, 0, dist_c, 1

        plt.quiver(x0, y0, dx, dy, scale_units='xy', angles='xy', scale=1, color=color)

    plt.scatter(all_nodes_position_x, all_nodes_position_y, color='blue')

    for i, txt in enumerate(all_nodes_position_labels):
        plt.annotate(txt, (all_nodes_position_x[i], all_nodes_position_y[i]))

    plt.show()
