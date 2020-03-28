# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################

from wcrt import WCRT
from pyro.control import nonlinear
###############################################################################

sys = WCRT()
ctl  = nonlinear.AdaptativeController_WCRT(sys)


#Param adapt-control
ctl.A[0] = 3
ctl.A[1] = 60
ctl.A[2] = 21
ctl.A[3] = 31
ctl.A[4] = 380
ctl.A[5] = 31

ctl.Kd[0,0] = 5
ctl.Kd[1,1] = 8
ctl.Kd[1,1] = 5

ctl.lam = 1.2

ctl.T[0,0] = 12
ctl.T[1,1] = 12
ctl.T[2,2] = 12
ctl.T[3,3] = 12
ctl.T[4,4] = 12
ctl.T[4,5] = 12

# Set Point
ctl.rbar = np.array([0,0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation

cl_sys.x0  = np.array([0,1,0,0,0,0])
tf = 12
n = tf*1000 + 1
cl_sys.compute_trajectory(tf, n, 'euler')
cl_sys.plot_trajectory()
cl_sys.animate_simulation(is_3d = True)

