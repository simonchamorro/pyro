#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import DrillingRobotOnJig
from chas2436      import CustomDrillingController


# Model dynamique du robot
sys = DrillingRobotOnJig()

# Controller
ctl = CustomDrillingController( ) # Empty do nothing controller template

# Closed-loop dynamic
clsys = ctl + sys

# États initiaux
#clsys.x0 =  np.array([0.7,1.45,-1.4,0,0,0]) #  Tombe dans le trou
clsys.x0 =  np.array([0,1.4,-1.3,0,0,0]) #

# Simulation
tf = 6
clsys.compute_trajectory( tf )
clsys.plot_trajectory('x')
clsys.plot_trajectory('u')
clsys.animate_simulation( is_3d = True )

# Exemple extraction des données pour analyse
q_traj   = clsys.traj.x[:,0:3]  # Trajectoire des angles du robot
dq_traj  = clsys.traj.x[:,3:6]  # Trajectoire des vitesses du robot
tau_traj = clsys.traj.u         # Trajectoire des couples du robot