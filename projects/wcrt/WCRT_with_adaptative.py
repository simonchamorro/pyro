# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from wcrt import WCRT 
from projects.adative_controllers import adaptive_computed_torque
###############################################################################

sys = WCRT()
ctl  = adaptive_computed_torque.AdaptativeController_WCRT(sys)


#Param adapt-control
ctl.A[0] = 4
ctl.A[1] = 5
ctl.A[2] = 3
ctl.A[3] = -10
ctl.A[4] = 6
ctl.A[5] = 20
ctl.A[6] = 10


ctl.Kd[0,0] = 7
ctl.Kd[1,1] = 7
ctl.Kd[2,2] = 7

ctl.lam = 1.5

ctl.T[0,0] = 8
ctl.T[1,1] = 8
ctl.T[2,2] = 8
ctl.T[3,3] = 8
ctl.T[4,4] = 8
ctl.T[5,5] = 8
ctl.T[6,6] = 8

# Set Point
ctl.rbar = np.array([0,0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation

cl_sys.x0  = np.array([1,1,0,0,0,0])
tf = 12
n = tf*1000 + 1
cl_sys.compute_trajectory(tf, n, 'euler')
cl_sys.plot_trajectory()
cl_sys.animate_simulation(is_3d = True)

