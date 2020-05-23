#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import DrillingRobotOnJig


# Model cinématique du robot
sys = DrillingRobotOnJig()

sys.x0 =  np.array([0.7,1.4,-1.3,0,0,0]) #

# Simulation
sys.compute_trajectory( 2 )
sys.plot_trajectory('x')
sys.animate_simulation( is_3d = True )