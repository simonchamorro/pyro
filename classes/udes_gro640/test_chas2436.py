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

def test_part1():
    r = 1
    d = 1 
    theta = np.pi
    alpha = np.pi
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(dh2T( r , d , theta, alpha ))

    r = np.array([1, 1, 1])
    d = np.array([1, 1, 1])
    theta = np.array([0, np.pi, np.pi])
    alpha = np.array([0, np.pi, 0])
    print(dhs2T( r , d , theta, alpha ))


def test_part2():
    pass


def test_part3():
    pass


def test_part4():
    pass



test_part1()