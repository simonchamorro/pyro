# -*- coding: utf-8 -*-
"""
Created on March 20 2020

@author: Pierre
"""
###############################################################################
import numpy as np
import math
###############################################################################

from wcrt import WCRT 
from projects.adative_controllers import adaptive_computed_torque

###############################################################################
pi = math.pi

sys = WCRT()

ctl  = adaptive_computed_torque.AdaptativeController_WCRT(sys)

#Param Wcrt
sys.d1 = 0
sys.d2 = 0
sys.d3 = 0
sys.m3 = 1.5
sys.lc2 = 0.8

#Param adapt-control
ctl.A[0] = 5
ctl.A[1] = 5
ctl.A[2] = 5
ctl.A[3] = 0
ctl.A[4] = 5
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
ctl.rbar = np.array([0,-pi/4,pi/2])

# New cl-dynamic
cl_sys = ctl + sys
#cl_sys = sys

# Simultation

cl_sys.x0  = np.array([pi/3,1,0,0,0,0])
tf = 8
n = tf*1000 + 1
cl_sys.compute_trajectory(tf, n, 'euler')
cl_sys.plot_trajectory()
cl_sys.animate_simulation(is_3d = True)

