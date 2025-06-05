import abc
import collections
import dataclasses
import itertools
import math
import typing

import numpy as np
import pandas as pd

from PIL import Image


def load_image(filename: str, mode: str = "RGB"):
    with Image.open(filename) as img:
        im = img.convert(mode)
        return im, np.asarray(im)



class PointData:
    def __init__(self, position, value):
        self.position = position
        self.value = value

    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]
    
    def __hash__(self):
        return hash((self.position, self.value))
    
    def __str__(self):
        return f"({self.position}, {self.value})"

    def __repr__(self):
        return f"({self.position}, {self.value})"


class AxisSegmentation:
    def __init__(self, index):
        self.__index = index
        self.__clusters = {}
        
    def append(self, point: PointData):
        a, b = point.position
        
        self.__clusters[b] = point
    
    def sort(self):
        return list(self.__clusters.values())
    
    @property
    def index(self):
        return self.__index
    
    @property
    def clusters(self):
        return self.__clusters
    
    def __iter__(self):
        return iter(self.__clusters)
    
    def __len__(self):
        return len(self.__clusters)


class TreeClassify:
    def __init__(self, size: tuple[int, int]):
        self.points = []
        self.__segmentation = {}

        width, height = size

        for c in range(width):
            self.__segmentation[c] = AxisSegmentation(c)
        
    def insert(self, point):
        self.points.append(point)
        
    def sort(self):
        for point in self.points:
            a, b = point.position
            self.__segmentation[a].append(point)
        
        clusters = []
        group = []
        
        for index, cluster in self.__segmentation.items():
            if len(cluster) <= 0:
                clusters.append(group)
                group = []
            else:
                group += cluster.sort()
        
        return [*filter(lambda a: len(a) > 0, clusters)]
    

if __name__ == "__main__":
    img, img_arr = load_image("resources/b-source-0.bmp", "I")
    pixels = []

    for y, row in enumerate(img_arr):
        for x, p in enumerate(row):
            pixels.append(PointData((x, y), p))

    mask_a, mask_b = [], []

    for p in pixels:
        if p.value != 0:
            mask_a.append(p)
        else:
            mask_b.append(p)

    width, height = img_arr.shape
    segmentation = TreeClassify((width, height))
    
    [segmentation.insert(p) for p in mask_b]

    clusters = segmentation.sort()

    for i, cluster in enumerate(clusters):
        print(f"Cluster[{i}]: {cluster}")
    