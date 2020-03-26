# -*- coding: utf-8 -*-
"""
Created on 19/11/2019

@author: Pierre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
from pyro.control import nonlinear
###############################################################################

sys = pendulum.DoublePendulum()
ctl = nonlinear.AdaptativeController_2(sys)

sys.m1 = 1

ctl.A[0] = 0
ctl.A[1] = 0
ctl.A[2] = 0
ctl.A[3] = 0
ctl.A[4] = 0

ctl.Kd[0,0] = 2
ctl.Kd[1,1] = 5
ctl.lam = 1
ctl.T[0,0] = 12
ctl.T[1,1] = 12
ctl.T[2,2] = 12
ctl.T[3,3] = 12
ctl.T[4,4] = 12
# Set Point
ctl.rbar = np.array([0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0  = np.array([-3.14,0,0,0])
tf = 12
n = tf*1000 + 1
cl_sys.compute_trajectory(tf, n, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()