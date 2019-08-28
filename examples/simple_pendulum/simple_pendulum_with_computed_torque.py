# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
from pyro.control import nonlinear
###############################################################################

sys  = pendulum.SinglePendulum()
ctl  = nonlinear.ComputedTorqueController( sys )

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([-2,0])
sim = cl_sys.compute_trajectory(x0=x_start)
cl_sys.plot_phase_plane_trajectory(sim)
cl_sys.plot_trajectory(sim, 'xu')
cl_sys.animate_simulation(sim)