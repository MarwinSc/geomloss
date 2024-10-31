import logging
from abc import ABC, abstractmethod
import numpy as np
import json
import struct
from util.timer import Timer

class Octree:

    def __init__(self, points, leaf_points, colors=None):

        self.num_leaf_points = leaf_points # stopage criterion

        self.node_count = [0] # nodes per level
        self.leaf_count = 0
        self.knot_count = 0
        self.point_count = len(points)

        self.colors = colors

        timer = Timer("Octree Creation:")
        bounds = np.array([[np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])],
                           [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]], dtype=np.float32)
        self.root = Root(points, colors, bounds, None, 0, self)

        timer.toc()
        
        # to keep track of references when writing the file
        self.hierarchy_id = 0
        self.point_id = 1

    def write(self, filename, as_bytes=False):
        timer = Timer("Octree Write:")
        if as_bytes:
            with (open(f"{filename}_hierarchy.bin", 'wb') as hierarchy_file,
                  open(f"{filename}_metadata.bin", 'wb') as metadata_file,
                  open(f"{filename}_points.bin", 'wb') as points_file):
                self.root.write(hierarchy_file, metadata_file, points_file, as_bytes=True)
        else:
            with (open(f"{filename}_hierarchy.txt", 'w', encoding='utf8') as hierarchy_file,
                  open(f"{filename}_metadata.txt", 'w', encoding='utf8') as metadata_file,
                  open(f"{filename}_points.txt", 'w', encoding='utf8') as points_file):
                self.root.write(hierarchy_file, metadata_file, points_file)
        timer.toc()

    def to_list(self):
        timer = Timer("Octree to list")

        self.hierarchy_id = 0
        self.point_id = 1
        self.hierarchy_pointer = 0
        
        self.hierarchy = np.zeros(self.knot_count * 8, dtype=np.uint32)
        self.bounds = np.zeros((self.knot_count + self.leaf_count, 6))
        self.metadata = np.zeros((self.knot_count + self.leaf_count, 5), dtype=np.uint32)
        self.points = np.zeros((self.point_count, 3))
        self.colors = np.zeros((self.point_count, 4))
        self.root.to_list()
        
        timer.toc()
        
        return self.hierarchy, self.bounds, self.metadata, self.points, self.colors


class Node:
    def __init__(self, bounds, nodeindex, level, octree):
        self.bounds = bounds
        self.isleaf = False
        self.nodeindex = nodeindex  # (x, y, z) 0 for positive 1 for negative
        self.level = level
        self.octree = octree

        self.extent = np.array([self.bounds[1, 0] - self.bounds[0, 0],
                                       self.bounds[1, 1] - self.bounds[0, 1],
                                       self.bounds[1, 2] - self.bounds[0, 2]], dtype=np.float32)

        self.mid = np.array([self.bounds[0, 0] + self.extent[0] / 2,
                                    self.bounds[0, 1] + self.extent[1] / 2,
                                    self.bounds[0, 2] + self.extent[2] / 2], dtype=np.float32)
        
        if len(octree.node_count) <= level:
            octree.node_count.append(1)
        else:
            octree.node_count[level] += 1


class Leaf(Node):
    def __init__(self, points, colors, bounds, nodeindex, level, octree):
        super().__init__(bounds, nodeindex, level, octree)
        self.isleaf = True
        self.points = points
        self.colors = colors
        octree.leaf_count += 1

    def __len__(self):
        if self.points is None:
            return 0
        else:
            return self.points.shape[0]

    def get_points(self):
        return self.points
        
    def to_list(self):

        # if leaf is not empty
        if self.points is not None and len(self) > 0:

            # used in the parent knot
            self.hierarchy_ref = self.octree.hierarchy_id
            self.octree.hierarchy_id += 1
            self.point_ref = 0

            self.point_ref = self.octree.point_id
            self.octree.point_id += len(self)

            self.octree.points[self.point_ref:self.point_ref+len(self), :] = self.points
            if self.colors is not None:
                self.octree.colors[self.point_ref:self.point_ref+len(self), :] = self.colors

            # isleaf as own list?                                   0           1               2                       3              4
            self.octree.metadata[self.hierarchy_ref, :] = np.r_[len(self), self.isleaf, self.point_ref+len(self), self.point_ref, self.level]
            self.octree.bounds[self.hierarchy_ref, :] = self.bounds.flatten()
        
    def write(self, hierarchyfile, metadatafile, pointfile, as_bytes=False):
        # used in the parent knot
        self.hierarchy_ref = self.octree.hierarchy_id
        self.octree.hierarchy_id += 1

        self.point_ref = 0
        # if leaf is not empty
        if self.points is not None:
            self.point_ref = self.octree.point_id
            self.octree.point_id += len(self)
            for i, point in enumerate(self.points):
                if as_bytes:
                    for value in point:
                        pointfile.write(struct.pack('<f', value))
                    if self.colors is not None:
                        for value in self.colors[i]:
                            pointfile.write(struct.pack('<f', value))
                else:
                    pointfile.write(" ".join([str(ii) for ii in point]) + "\n")
                    if self.colors is not None:
                        pointfile.write(" ".join([str(ii) for ii in self.colors[i]]) + "\n")

        if as_bytes:
            for value in self.bounds.flatten():
                bs = struct.pack('<f', value)
                metadatafile.write(bs)
            metadatafile.write(struct.pack('<I', len(self)))
            metadatafile.write(struct.pack('<I', 1 if self.isleaf else 0))
            metadatafile.write(struct.pack('<I', self.point_ref))
        else:
            metadatafile.write(" ".join([str(i) for i in self.bounds.flatten()]) + "\n")
            metadatafile.write(str(len(self)) + "\n")
            metadatafile.write(str(1 if self.isleaf else 0) + "\n")
            metadatafile.write(str(self.point_ref) + "\n")


childindex_lut = {
    "000": 0,   # lower, left, front
    "100": 1,   # lower, right, front
    "010": 2,   # upper, left, front
    "110": 3,   # upper, right, front
    "001": 4,   # lower, left, back
    "101": 5,   # lower, right, back
    "011": 6,   # upper, left, back
    "111": 7,   # upper, right, back
}

class Knot(Node):

    def __init__(self, points, colors, bounds, nodeindex, level, octree):
        super().__init__(bounds, nodeindex, level, octree)
        self.children = None
        self.total_points = None
        self.setup_children(points, colors)
        octree.knot_count += 1


    def __len__(self):
        return self.total_points
    
    def add(self, points, colors, bounds, nodeindex):

        # check which points are inside this knot
        mask = np.logical_and(
            np.logical_and(
                np.logical_and(points[:, 0] >= bounds[0, 0], points[:, 0] < bounds[1, 0]),
                np.logical_and(points[:, 1] >= bounds[0, 1], points[:, 1] < bounds[1, 1])
            ),
            np.logical_and(points[:, 2] >= bounds[0, 2], points[:, 2] < bounds[1, 2])
        )

        num = np.count_nonzero(mask)

        points = points[mask]
        colors = colors[mask]
        
        if num < self.octree.num_leaf_points:
            self.children.append(Leaf(points, colors, bounds, nodeindex, self.level + 1, self.octree))
        else:
            self.children.append(Knot(points, colors, bounds, nodeindex, self.level + 1, self.octree))
        return num


    def setup_children(self, points, colors):
        self.children = []  
        self.total_points = 0

        # left, lower, front
        bounds = np.array([[self.bounds[0, 0], self.bounds[0, 1], self.bounds[0, 2]],
                           [self.mid[0], self.mid[1], self.mid[2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "000")

        # right, lower, front
        bounds = np.array([[self.mid[0], self.bounds[0, 1], self.bounds[0, 2]],
                           [self.bounds[1, 0], self.mid[1], self.mid[2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "100")

        # left, upper, front
        bounds = np.array([[self.bounds[0, 0], self.mid[1], self.bounds[0, 2]],
                           [self.mid[0], self.bounds[1, 1], self.mid[2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "010")

        # right, upper, front
        bounds = np.array([[self.mid[0], self.mid[1], self.bounds[0, 2]],
                           [self.bounds[1, 0], self.bounds[1, 1], self.mid[2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "110")

        # left, lower, back
        bounds = np.array([[self.bounds[0, 0], self.bounds[0, 1], self.mid[2]],
                           [self.mid[0], self.mid[1], self.bounds[1, 2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "001")

        # right, lower, back
        bounds = np.array([[self.mid[0], self.bounds[0, 1], self.mid[2]],
                           [self.bounds[1, 0], self.mid[1], self.bounds[1, 2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "101")

        # left, upper, back
        bounds = np.array([[self.bounds[0, 0], self.mid[1], self.mid[2]],
                           [self.mid[0], self.bounds[1, 1], self.bounds[1, 2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "011")

        # right, upper, back
        bounds = np.array([[self.mid[0], self.mid[1], self.mid[2]],
                           [self.bounds[1, 0], self.bounds[1, 1], self.bounds[1, 2]]], dtype=np.float32)
        self.total_points += self.add(points, colors, bounds, "111")

    def get_nodeindex(self, point):
        nodeindex = ""
        for i in range(3):
            if self.bounds[0, i] <= point[i] <= self.mid[i]:
                nodeindex += "0"
            elif self.mid[i] < point[i] <= self.bounds[1, i]:
                nodeindex += "1"
            else:
                raise Exception("Not in the node.")
        return nodeindex
    
    def to_list(self):

        # bottom up write the leafs first
        for child in self.children:
            child.to_list()

        # used in the parent knot
        self.hierarchy_ref = self.octree.hierarchy_id
        self.octree.hierarchy_id += 1

        point_range_from = []
        point_range_to = []
        for i, child in enumerate(self.children):
            if len(child) > 0:
                point_range_from.append(child.point_ref)
                point_range_to.append(child.point_ref + len(child))
                self.octree.hierarchy[self.octree.hierarchy_pointer + i] = child.hierarchy_ref

        self.point_ref = np.min(np.r_[point_range_from])

        self.octree.hierarchy_pointer += 8

        # isleaf as own list?
        self.octree.metadata[self.hierarchy_ref, :] = np.r_[len(self), self.isleaf, self.hierarchy_ref, self.point_ref, self.level]
        self.octree.bounds[self.hierarchy_ref, :] = self.bounds.flatten()


    def write(self, hierarchyfile, metadatafile, pointfile, as_bytes=False):

        # bottom up write the leafs first
        for child in self.children:
            child.write(hierarchyfile, metadatafile, pointfile, as_bytes=as_bytes)

        self.hierarchy_ref = self.octree.hierarchy_id
        self.octree.hierarchy_id += 1

        if as_bytes:
            for child in self.children:
                hierarchyfile.write(struct.pack('<I', child.hierarchy_ref))

            for value in self.bounds.flatten():
                bs = struct.pack('<f', value)
                metadatafile.write(bs)
            metadatafile.write(struct.pack('<I', len(self)))
            metadatafile.write(struct.pack('<I', 1 if self.isleaf else 0))
            metadatafile.write(struct.pack('<I', self.hierarchy_ref))
        else:
            for child in self.children:
                hierarchyfile.write(str(child.hierarchy_ref) + "\n")
            hierarchyfile.write("\n")

            metadatafile.write(" ".join([str(i) for i in self.bounds.flatten()]) + "\n")
            metadatafile.write(str(len(self)) + "\n")
            metadatafile.write(str(1 if self.isleaf else 0) + "\n")
            metadatafile.write(str(self.hierarchy_ref) + "\n")


class Root(Knot):
    def __init__(self, points, colors, bounds, nodeindex, level, octree):
        super().__init__(points, colors, bounds, nodeindex, level, octree)
        octree.knot_count += 1

