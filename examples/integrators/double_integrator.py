# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:17:36 2018

@author: Alexandre
"""

###############################################################################
import numpy as np

###############################################################################
from pyro.dynamic  import integrator
from pyro.analysis import costfunction
    
###################################
# Simple integrator
###################################

di = integrator.DoubleIntegrator()

di.ubar = np.array([1]) # constant input = 1


###################################
# Analysis
###################################
    
# Phase plane behavior test
di.plot_phase_plane()

# Simulation
sim = di.compute_trajectory(x0=np.array([0,0]))
di.plot_trajectory(sim)
di.plot_trajectory(sim, 'y')


# Cost computing

# Weights for quadratic cost function
q, r, v = np.ones(di.n), np.ones(di.m), np.zeros(di.p)
qcf = costfunction.QuadraticCostFunction(q, r, v)

sim_with_quad_cost = qcf.eval(sim)
di.plot_trajectory(sim_with_quad_cost, 'xuj')

# Time cost
tcf = costfunction.TimeCostFunction( di.xbar )
sim_with_time_cost = tcf.eval(sim)
di.plot_trajectory(sim_with_time_cost, 'j')

# Phase plane trajectory
di.plot_phase_plane_trajectory( sim )