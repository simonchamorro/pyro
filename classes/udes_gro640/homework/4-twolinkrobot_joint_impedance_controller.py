# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
@author: Simon Chamorro

Devoir #4 c) joint impedance
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control  import robotcontrollers
from pyro.dynamic  import manipulator
###############################################################################


class JointGravityPID(robotcontrollers.JointPID):

    def __init__(self, robot, dof = 1, kp = 1, ki = 0, kd = 0):
        self.robot = robot
        super().__init__(dof, kp, ki, kd)

    def c( self , y , r , t = 0 ):
        u = np.zeros(self.m) 
        
        # Ref
        q_d = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Error
        e  = q_d - q
        de =     - dq
        ie = self.e_int + e * self.dt # TODO use dynamic controller class
        
        # PIDs
        u = e * self.kp + de * self.kd + ie * self.ki

        # Add gravity compensation
        u += self.robot.g( q )
        
        # Memory
        self.e_int =  ie # TODO use dynamic controller class
        
        return u


torque_controlled_robot      = manipulator.TwoLinkManipulator()

# Target
q_desired = np.array([0.5,0.5])

# Joint PID

dof = 2

# Use JointGravityPID for gravity compensation 
# joint_pid      = robotcontrollers.JointPID( dof )
joint_pid      = JointGravityPID( torque_controlled_robot, dof=dof )

joint_pid.rbar = q_desired
joint_pid.kp   = np.array([10, 5 ])
joint_pid.kd   = np.array([ 1, 0 ])


# Closed-loops

robot_with_joint_pid    = joint_pid + torque_controlled_robot 

# Simulations
tf = 4
robot_with_joint_pid.x0 = np.array([0,0,0,0])
robot_with_joint_pid.compute_trajectory( tf )
# robot_with_joint_pid.plot_trajectory('xu')
robot_with_joint_pid.animate_simulation()
