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

# Dynamic model
torque_controlled_robot = manipulator.FiveLinkPlanarManipulator()
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )

# Controller
kinematic_controller = robotcontrollers.EndEffectorKinematicControllerWithNullSpaceTask( speed_controlled_robot )

# Main objective
kinematic_controller.rbar  = np.array([1.0,1.0])
kinematic_controller.gains = np.array([1.0,1.0])

# Secondary objective
kinematic_controller.q_d         = np.array([-1,-1,0,0,0])
kinematic_controller.gains_null  = np.array([100,100,0,0,0])


# Closed-loop dynamics
closed_loop_robot = kinematic_controller + speed_controlled_robot

# Simulation and plots
closed_loop_robot.x0        = np.array([0.1,0.1,0.1,0.1,0.1])

closed_loop_robot.plot_trajectory('x')
closed_loop_robot.animate_simulation()
