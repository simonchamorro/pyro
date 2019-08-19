# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import robotcontrollers
from pyro.analysis import simulation
###############################################################################

sys  = pendulum.SinglePendulum()

dof = 1

kp = 2 # 2,4
kd = 1 # 1
ki = 1

ctl  = robotcontrollers.JointPID( dof, kp , ki, kd)

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([0,0])

sim = cl_sys.compute_trajectory(x_start, tf=20, n=20001, solver='euler')
cl_sys.get_plotter().phase_plane_trajectory(sim, 0, 1)
cl_sys.get_plotter().plot(sim, 'xu')
cl_sys.get_animator().animate_simulation(sim)