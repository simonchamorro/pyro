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
ctl  = nonlinear.ComputedTorqueController( sys )

ctl.w0   = 1.5
ctl.zeta = 0.5 
ctl.rbar = np.array([0,0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
cl_sys.x0 = np.array([1,1,1,0,0,0])
cl_sys.compute_trajectory( 10 , 10001, 'euler')
cl_sys.plot_trajectory('x')
cl_sys.plot_trajectory('u')
cl_sys.animate_simulation( is_3d = True )