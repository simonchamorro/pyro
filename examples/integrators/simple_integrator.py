# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:13:40 2018

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

si = integrator.SimpleIntegrator()

si.ubar = np.array([1]) # constant input = 1

###################################
# Analysis
###################################

# Phase plane
si.plot_phase_plane(0,0) # only one state for two axis!

# Cost function with unit weights
qcf = costfunction.QuadraticCostFunction(
    np.ones(si.n),
    np.ones(si.m),
    np.zeros(si.p)
)

# Simulation
sim = si.compute_trajectory(np.array([2]), costfunc=qcf)

# Plot output
si.plot_trajectory(sim, 'xuj')
si.plot_phase_plane_trajectory(sim,0,0)