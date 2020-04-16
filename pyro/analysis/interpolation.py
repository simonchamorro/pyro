# -*- coding: utf-8 -*-
"""
@author: amy-f
"""
import math
import numpy as np
from numba import jit


# Numba implementations of 2D and 3D classes
def nearest_neighbor_2D(x_next, nodes_state, nodes_index, xgriddim, J):
    J_indices = find_indices(x_next, nodes_state, nodes_index, xgriddim, J)
    return J[J_indices[0]][J_indices[1]]


@jit(nopython=True, parallel=True)
def find_indices(x_next, nodes_state, nodes_index, xgriddim, J):
    nodes_dict = {}
    for i in range(len(nodes_state)):
        nodes_dict[(nodes_state[i][0], nodes_state[i][1])] = (nodes_index[i][0], nodes_index[i][1])

    # get scaled values from grid
    x_scale = xgriddim[0] / J.shape[0]
    y_scale = xgriddim[1] / J.shape[1]

    interpolated_x = x_next[0] / x_scale
    interpolated_y = x_next[1] / y_scale

    # find the closest neighbor
    neighbors = find_neighbor(nodes_state, interpolated_x, interpolated_y, xgriddim)

    distance = np.array([euclidian_distance_2D(interpolated_x, interpolated_y, n) for n in neighbors])
    closest = distance.argmin()

    # fetch and return J_value at this position
    closest_neighbor = (neighbors[closest][0], neighbors[closest][1])
    return nodes_dict.get(closest_neighbor)


@jit(nopython=True)
def find_neighbor(nodes_state, interpolated_x, interpolated_y, xgriddim):
    # get grid step and scale
    x_step = abs(nodes_state[xgriddim[0]][0] - nodes_state[0][0])
    y_step = abs(nodes_state[1][1] - nodes_state[0][1])

    neighbors = []
    for i in range(xgriddim[0] * xgriddim[1]):
        x = nodes_state[i][0]
        y = nodes_state[i][1]
        if interpolated_x - x_step <= x <= interpolated_x + x_step and interpolated_y - y_step <= y <= interpolated_y + y_step:
            neighbors.append((x, y))
    return neighbors


@jit(nopython=True)
def euclidian_distance_2D(x, y, neighbor):
    return math.sqrt((x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2)

# Numba implementations of 3D classes
def nearest_neighbor_3D(x_next, nodes_state, nodes_index, xgriddim, J):
    J_indices = find_indices_3D(x_next, nodes_state, nodes_index, xgriddim, J)
    return J[J_indices[0]][J_indices[1]][J_indices[2]]


@jit(nopython=True, parallel=True)
def find_indices_3D(x_next, nodes_state, nodes_index, xgriddim, J):
    nodes_dict = {}
    for i in range(len(nodes_state)):
        nodes_dict[(nodes_state[i][0], nodes_state[i][1], nodes_state[i][2])] = (nodes_index[i][0], nodes_index[i][1], nodes_index[i][2])

    # get scaled values from grid
    x_scale = xgriddim[0] / J.shape[0]
    y_scale = xgriddim[1] / J.shape[1]
    z_scale = xgriddim[2] / J.shape[2]

    interpolated_x = x_next[0] / x_scale
    interpolated_y = x_next[1] / y_scale
    interpolated_z = x_next[2] / z_scale

    # find the closest neighbor
    neighbors = find_neighbor_3D(nodes_state, interpolated_x, interpolated_y, interpolated_z, xgriddim)

    distance = np.array([euclidian_distance_3D(interpolated_x, interpolated_y, interpolated_z, n) for n in neighbors])
    closest = distance.argmin()

    # fetch and return J_value at this position
    closest_neighbor = (neighbors[closest][0], neighbors[closest][1], neighbors[closest][2])
    return nodes_dict.get(closest_neighbor)


@jit(nopython=True)
def find_neighbor_3D(nodes_state, interpolated_x, interpolated_y, interpolated_z, xgriddim):
    # get grid step and scale
    x_step = abs(nodes_state[xgriddim[0] * xgriddim[1]][0] - nodes_state[0][0])
    y_step = abs(nodes_state[xgriddim[1]][1] - nodes_state[0][1])
    z_step = abs(nodes_state[1][2] - nodes_state[0][2])

    neighbors = []
    for i in range(xgriddim[0] * xgriddim[1] * xgriddim[2]):
        x = nodes_state[i][0]
        y = nodes_state[i][1]
        z = nodes_state[i][2]
        if interpolated_x - x_step <= x <= interpolated_x + x_step \
                and interpolated_y - y_step <= y <= interpolated_y + y_step \
                and interpolated_z - z_step <= z <= interpolated_z + z_step:
            neighbors.append((x, y, z))
    return neighbors


@jit(nopython=True)
def euclidian_distance_3D(x, y, z, neighbor):
    return math.sqrt((x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2 + (z - neighbor[2]) ** 2)


class Interpolation:

    def __init__(self, sys, grid, J):
        self.grid = grid
        self.nodes_state = grid.nodes_state
        self.nodes_index = grid.nodes_index

        self.lower_bounds = sys.x_lb
        self.upper_bounds = sys.x_ub
        self.J = J

        # get grid step and scale
        self.x_step = abs(self.nodes_state[self.grid.xgriddim[0]][0] - self.nodes_state[0][0])
        self.y_step = abs(self.nodes_state[1][1] - self.nodes_state[0][1])

        # get scaled values from grid
        self.x_scale = self.grid.xgriddim[0] / self.J.shape[0]
        self.y_scale = self.grid.xgriddim[1] / self.J.shape[1]

    # Implement directly in child classes
    def nearest_neighbor(self, x_next):
        raise NotImplementedError

    def linear_interpolation(self, x_next):
        raise NotImplementedError

    def euclidian_distance(self, x, y, x_next):
        raise NotImplementedError

class Interpolation2D(Interpolation):

    def __init__(self, sys, grid, J):
        super().__init__(sys, grid, J)

    # Gets the nearest neighbor of point x_next in grid J to system grid
    def nearest_neighbor(self, x_next):
        self.nodes_dict = {}
        for i in range(len(self.nodes_state)):
            self.nodes_dict[tuple(self.nodes_state[i])] = tuple(self.nodes_index[i])

        interpolated_x = x_next[0] / self.x_scale
        interpolated_y = x_next[1] / self.y_scale

        # find the closest neighbor
        neighbors = [(x, y) for x, y in self.nodes_state
                     if interpolated_x - self.x_step <= x <= interpolated_x + self.x_step
                     and interpolated_y - self.y_step <= y <= interpolated_y + self.y_step]
        closest = np.argmin([self.euclidian_distance(interpolated_x, interpolated_y, n) for n in neighbors])

        # fetch and return J_value at this position
        J_indices = self.nodes_dict.get((neighbors[closest][0], neighbors[closest][1]))
        print(J_indices)
        return self.J[J_indices]

    def euclidian_distance(self, x, y, neighbor):
        return math.sqrt((x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2)


class Interpolation3D(Interpolation):
    def __init__(self,sys, grid_sys, J):
        super().__init__(sys, grid_sys, J)

    # Gets the nearest neighbor of point x_next in grid J to system grid
    def nearest_neighbor(self, x_next):
        self.nodes_dict = {}
        for i in range(len(self.nodes_state)):
            self.nodes_dict[tuple(self.nodes_state[i])] = tuple(self.nodes_index[i])

        interpolated_x = x_next[0] / self.x_scale
        interpolated_y = x_next[1] / self.y_scale

        # find the closest neighbor
        neighbors = [(x, y) for x, y in self.nodes_state
                     if interpolated_x - self.x_step <= x <= interpolated_x + self.x_step
                     and interpolated_y - self.y_step <= y <= interpolated_y + self.y_step]
        closest = np.argmin([self.euclidian_distance(interpolated_x, interpolated_y, n) for n in neighbors])

        # fetch and return J_value at this position
        J_indices = self.nodes_dict.get((neighbors[closest][0], neighbors[closest][1]))
        print(J_indices)
        return self.J[J_indices]

    def euclidian_distance(self, x, y, neighbor):
        return math.sqrt((x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2)

