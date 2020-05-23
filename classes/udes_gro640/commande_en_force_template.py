#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import DrillingRobotOnJig
from abcd1234      import CustomDrillingController


# Model dynamique du robot
sys = DrillingRobotOnJig()

# Controller
ctl = CustomDrillingController( )

# Closed-loop dynamic
clsys = ctl + sys

# États initiaux
clsys.x0 =  np.array([0.7,1.4,-1.3,0,0,0]) #
#clsys.x0 =  np.array([0,1.4,-1.3,0,0,0]) #

# Simulation
tf = 6
clsys.compute_trajectory( tf )
clsys.plot_trajectory('x')
clsys.plot_trajectory('u')
clsys.animate_simulation( is_3d = True )

# Exemple pour analyse
x = clsys.traj.x  # Trajectoire des états du robot
u = clsys.traj.u  # Trajectoire des inputs du robot