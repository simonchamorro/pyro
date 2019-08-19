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
#ctl  = nonlinear.ComputedTorqueController( sys )
ctl  = nonlinear.SlidingModeController( sys )

ctl.lam  = 5.0
ctl.gain = 5.0

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([0,0])
sim = cl_sys.compute_trajectory(x_start, tf=10, n=1001, solver='euler')
cl_sys.get_plotter().phase_plane_trajectory_closed_loop(sim, 0, 1)
cl_sys.get_plotter().phase_plane_trajectory(sim, 0, 1)
cl_sys.get_animator().animate_simulation(sim)