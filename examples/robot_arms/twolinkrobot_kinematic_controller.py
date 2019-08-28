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

torque_controlled_robot = manipulator.TwoLinkManipulator()
speed_controlled_robot  = manipulator.SpeedControlledManipulator( 
                                             torque_controlled_robot )


kinematic_controller = robotcontrollers.EndEffectorKinematicController( speed_controlled_robot , 1 )
kinematic_controller.rbar = np.array([0.5,0.5])
    
closed_loop_robot = kinematic_controller + speed_controlled_robot
    
x0        = np.array([-0.5,0.2])
    
sim = closed_loop_robot.compute_trajectory( x0, 5 )
closed_loop_robot.plot_trajectory(sim, 'xu')
closed_loop_robot.animate_simulation(sim)
