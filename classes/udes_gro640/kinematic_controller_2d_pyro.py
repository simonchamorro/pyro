#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 28

@author: Simon Chamorro CHAS2436
------------------------------------

Devoir #3 d) using pyro

"""

import numpy as np
from pyro.control import robotcontrollers
from pyro.dynamic import manipulator


class CircleController(robotcontrollers.EndEffectorKinematicController):

    def c(self, y, r, t=0):
        r_d = np.array([0.5*np.cos(0.5*t*t), 0.5*np.sin(0.5*t*t)])
        return super().c(y, r_d, t=0)


# Dynamic model (inputs are motor torques)
torque_controlled_robot    = manipulator.TwoLinkManipulator()
torque_controlled_robot.l1 = 1
torque_controlled_robot.l2 = 1

# Kinematic only model (inputs are motor velocities)
speed_controlled_robot  = manipulator.SpeedControlledManipulator.from_manipulator( torque_controlled_robot )


# Kinematic effector position controller
kinematic_controller = CircleController( speed_controlled_robot )

# Controller params
kinematic_controller.gains = np.array([100,100]) 
kinematic_controller.rbar  = np.array([0.5,0.5]) # Goal position


# Closed-loop system
closed_loop_robot = kinematic_controller + speed_controlled_robot

# Initial position of the robot
closed_loop_robot.x0        = np.array([0.25268, 2.6362])

# Simulation
closed_loop_robot.animate_simulation()