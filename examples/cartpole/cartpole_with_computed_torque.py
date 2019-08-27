# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
from pyro.control  import nonlinear
###############################################################################


sys  = cartpole.RotatingCartPole()

ctl  = nonlinear.ComputedTorqueController( sys )

ctl.w0   = 1.0
ctl.zeta = 0.7

# goal
ctl.rbar = np.array([0,0])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start = np.array([-3.14,-3.14,0,0])
sim = cl_sys.compute_trajectory(x0=x_start)
cl_sys.get_plotter().phase_plane_trajectory(sim)
cl_sys.get_plotter().plot(sim, 'xu')
cl_sys.animate_simulation(sim, time_factor_video=1.0, is_3d=True)