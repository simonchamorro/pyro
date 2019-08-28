# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control  import robotcontrollers
from pyro.dynamic  import manipulator
###############################################################################

torque_controlled_robot      = manipulator.TwoLinkManipulator()
torque_controlled_robot.ubar = np.array([2,0.8]) # constant inputs

x0        = np.array([0,1,0,0])

tc_sim = torque_controlled_robot.compute_trajectory(x0)
torque_controlled_robot.animate_simulation(tc_sim)
torque_controlled_robot.plot_trajectory(tc_sim, 'xu')

speed_controlled_robot      = manipulator.SpeedControlledManipulator( 
                                             torque_controlled_robot )
speed_controlled_robot.ubar = np.array([2,0.8]) # constant inputs
    
x0        = np.array([0,1])

sc_sim = speed_controlled_robot.compute_trajectory(x0)
speed_controlled_robot.animate_simulation(sc_sim)
speed_controlled_robot.plot_trajectory(sc_sim, 'xu')