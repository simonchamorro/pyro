# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""

from pathlib import Path
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import nonlinear
from pyro.planning import plan
from pyro.analysis import Trajectory
###############################################################################

sys  = pendulum.SinglePendulum()



# Controller

this_script_dir = Path(__file__).parent
traj_file = this_script_dir.joinpath('pendulum_rrt.npy')
traj = Trajectory.load(str(traj_file))

#ctl  = nonlinear.ComputedTorqueController( sys , traj )
ctl  = nonlinear.SlidingModeController( sys , traj )

ctl.lam  = 5
ctl.gain = 2

# goal
ctl.rbar = np.array([-3.14])

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([0.1,0])
sim = cl_sys.compute_trajectory(x_start, tf=5, n=10001, solver='euler')
cl_sys.plot_trajectory(sim)
cl_sys.animate_simulation(sim)