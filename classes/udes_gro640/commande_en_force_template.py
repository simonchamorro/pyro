#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import Robot1, Robot2, Robot3


# Model cinématique du robot
sys = Robot2()

sys.x0 =  np.array([0,-1.3,2.6,0,0,0]) #

# Simulation
sys.compute_trajectory()
sys.plot_trajectory( plot='x' )
sys.animate_simulation( is_3d = True )