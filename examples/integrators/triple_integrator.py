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

ti = integrator.TripleIntegrator()

ti.ubar = np.array([1]) # constant input = 1

###################################
# Analysis
###################################

# Simulation

qcf = costfunction.QuadraticCostFunction(
    np.ones(ti.n),
    np.ones(ti.m),
    np.zeros(ti.p)
)

sim = ti.compute_trajectory(x0=np.array([2,0,0]), costfunc=qcf)

ti.plot_trajectory(sim, 'xuj')
ti.plot_phase_plane_trajectory(sim, 0, 1)