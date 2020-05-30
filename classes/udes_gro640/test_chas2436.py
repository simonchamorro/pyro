#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26

@author: Simon Chamorro CHAS2436
------------------------------------


Tests for GRO640 

"""

import numpy as np

from gro640_robots import *
from chas2436      import *

def compare_arrays(a, b, e=0.0001):
    equal = True
    for i in range(len(a)):
        if (abs(a[i] - b[i]) > e):
            equal = False
    return equal


def test_part1():
    # Test func dh2T
    r = 1
    d = 1 
    theta = np.pi
    alpha = np.pi
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(dh2T( r , d , theta, alpha ))

    # Test func dhs2T
    r = np.array([1, 2, 3, 4, 5])
    d = np.array([1, 2, 3, 4, 5])
    theta = np.array([1, 2, 3, 4, 5])
    alpha = np.array([1, 2, 3, 4, 5])
    print(dhs2T( r , d , theta, alpha ))

    q = [0, 0, 0, 0, 0, 0]
    assert compare_arrays(f(q), [-0.033, 0, 0.655])
    print(f(q))

    q = [0, 0, 0, 0, 0]
    x = [0]
    q = [ang*np.pi/180 for ang in q] + x
    print(f(q))


def test_part2():
    # See commande_en_position_solution_template.py
    pass


def test_part3():
    pass


def test_part4():
    pass



test_part1()