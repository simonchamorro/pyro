# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import robotcontrollers, linear
from pyro.analysis import simulation
###############################################################################


class SinglePendulum1out(pendulum.SinglePendulum):
    def __init__(self):
        super().__init__()
        self.p = 1

    def h(self, x, u, t):
        return super().h(x, u, t)[0][np.newaxis]

sys  = SinglePendulum1out()
dof = 1

kp = 2 # 2,4
kd = 1 # 1
ki = 1

ctl  = robotcontrollers.JointPID( dof, kp , ki, kd)
ctl = linear.PIDController(kp, ki, kd)

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([0,0])

sim = cl_sys.compute_trajectory(x_start, tf=20, n=20001, solver='euler')
cl_sys.plot_phase_plane_trajectory(sim, 0, 1)
cl_sys.plot_trajectory(sim, 'xu')
cl_sys.get_animator().animate_simulation(sim)