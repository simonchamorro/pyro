# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""

#import matplotlib
#matplotlib.use('Qt4Agg')

###############################################################################
import numpy as np
###############################################################################
from pyro.analysis import costfunction
from pyro.control import controller
from pyro.dynamic import pendulum
###############################################################################


import matplotlib.pyplot as plt

from pyro.planning import discretizer, valueiteration

sys = pendulum.DoublePendulum()

# Discrete world
grid_sys = discretizer.GridDynamicSystem( sys )

# Cost Function
qcf = costfunction.QuadraticCostFunction(
    np.ones(sys.n),
    np.ones(sys.m),
    np.ones(sys.p)
)

qcf.xbar = np.array([ -3.14 , 0 ]) # target
qcf.INF  = 10000

# VI algo
vi = valueiteration.ValueIteration_ND( grid_sys , qcf )

vi.initialize()
# vi.load_data('simple_pendulum_vi')
vi.compute_steps()
#vi.load_data()
vi.assign_interpol_controller()
vi.plot_policy(0)
vi.plot_cost2go()
vi.save_data('double_pendulum_vi')

#asign controller
cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )

# Simulation and animation
x0   = [0,0]
tf   = 10
sim = cl_sys.compute_trajectory(x0, tf, costfunc=qcf)
cl_sys.get_plotter().plot(sim, 'xuj')
cl_sys.get_animator().animate_simulation(sim)
