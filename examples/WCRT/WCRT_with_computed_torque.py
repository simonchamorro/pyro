# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import WCRT
from pyro.control import nonlinear
###############################################################################

sys = WCRT.B3DDL()
ctl  = nonlinear.ComputedTorqueController( sys )

ctl.w0   = 1.5
ctl.zeta = 0.5 
ctl.rbar = np.array([0,0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([1,0,1,0,0,0])
cl_sys.plot_trajectory( x_start , 10 , 10001, 'euler')
cl_sys.sim.phase_plane_trajectory(0,2)
cl_sys.animate_simulation()