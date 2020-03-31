# -*- coding: utf-8 -*-
"""
@author: amy-f
"""
import math

import numpy as np
from numba import jit

# Numba implementations of 2D classes
@jit(nopython=True)
def nearest_neighbor_2D(x_next, nodes_state, nodes_index, xgriddim, J):

    # get grid step and scale
    x_step = abs(nodes_state[xgriddim[0]][0] - nodes_state[0][0])
    y_step = abs(nodes_state[1][1] - nodes_state[0][1])

    # get scaled values from grid
    x_scale = xgriddim[0] / J.shape[0]
    y_scale = xgriddim[1] / J.shape[1]

    nodes_dict = {}
    for i in range(len(nodes_state)):
        nodes_dict[(nodes_state[i][0], nodes_state[i][1])] = (nodes_index[i][0], nodes_index[i][1])

    interpolated_x = x_next[0] / x_scale
    interpolated_y = x_next[1] / y_scale

    # find the closest neighbor
    neighbors = [(x, y) for x, y in nodes_state
                 if interpolated_x - x_step <= x <= interpolated_x + x_step
                 and interpolated_y - y_step <= y <= interpolated_y + y_step]
    distance = [euclidian_distance_2D(interpolated_x, interpolated_y, n) for n in neighbors]
    closest = np.argmin(distance)

    # fetch and return J_value at this position
    J_indices = nodes_dict.get((neighbors[closest][0], neighbors[closest][1]))
    return J[J_indices]

@jit(nopython=True)
def euclidian_distance_2D(x, y, neighbor):
    return math.sqrt((x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2)


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
    @jit(nopython=True)
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
        return self.J[J_indices]

    def euclidian_distance(self, x, y, neighbor):
        return math.sqrt((x - neighbor[0]) ** 2 + (y - neighbor[1]) ** 2)


class Interpolation3D(Interpolation):
    def __init__(self,sys, grid_sys, J):
        super().__init__(sys, grid_sys, J)

    def nearest_neighbor(self, x_next):
        raise NotImplementedError

