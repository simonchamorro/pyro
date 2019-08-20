# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

from pathlib import Path

import numpy as np

from pyro.dynamic  import pendulum
from pyro.planning import OpenLoopController


# Dynamic system
sys  = pendulum.SinglePendulum()

# Openloop controller loading RRT solution
this_script_dir = Path(__file__).parent
traj_file = this_script_dir.joinpath('pendulum_rrt.npy')
ctl = OpenLoopController.load_from_file(str(traj_file))

# Closing the loop
cl_sys = ctl + sys

# Simulation
x_start = np.array([0.1,0])

# Stop the simulation before the end of the recorded trajectory because the ODE
# solver needs to evaluate the function at points after tf
cl_sys.plot_trajectory(  x_start, tf=(ctl.time_final - 0.01) )

cl_sys.animate_simulation()