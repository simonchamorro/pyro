#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:16:59 2019

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle
###############################################################################

# Vehicule dynamical system
sys = vehicle.KinematicCarModel()

# Set default wheel velocity and steering angle
sys.ubar = np.array([2,0.1])

# Plot open-loop behavior
sys.plot_trajectory( np.array([0,0,0]) , 10 )

# Animate the simulation
sys.animate_simulation( )