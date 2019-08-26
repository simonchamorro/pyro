# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import cartpole
###############################################################################

sys  = cartpole.RotatingCartPole()

# Simultation
x_start = np.array([0,0.1,0,0])
sim = sys.compute_trajectory(x0=x_start)
sys.get_plotter().phase_plane_trajectory(sim)
sys.get_plotter().plot(sim, 'xu')
sys.animate_simulation(sim, time_factor_video=1.0, is_3d=True)