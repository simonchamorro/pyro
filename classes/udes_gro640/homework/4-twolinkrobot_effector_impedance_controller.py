# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:44 2019

@author: Alexandre
@author: Simon Chamorro

Devoir #4 c) end effector impedance
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.control  import robotcontrollers
from pyro.dynamic  import manipulator
###############################################################################


class EndEffectorGravityPID(robotcontrollers.EndEffectorPID):

    def __init__(self, robot, kp = 1, ki = 0, kd = 0):
        self.robot = robot
        super().__init__(robot, kp, ki, kd)

    def c( self , y , r , t = 0 ):
        u = np.zeros(self.m) 
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        dr          = np.dot( J , dq )
        
        # Error
        e  = r_desired - r_actual
        de =           - dr
        ie = self.e_int + e * self.dt # TODO use dynamic controller class
        
        # Effector space PID
        f = e * self.kp + de * self.kd + ie * self.ki
        
        # From effector force to joint torques
        u = np.dot( J.T , f )
        
        # Add gravity compensation
        u += self.robot.g( q ) 

        # Memory
        self.e_int =  ie # TODO use dynamic controller class
        
        return u


torque_controlled_robot      = manipulator.TwoLinkManipulator()

# Target
q_desired = np.array([0.5,0.5])
r_desired = torque_controlled_robot.forward_kinematic_effector( q_desired )


# Effector PID 

model = torque_controlled_robot

# Use EndEffectorGravityPID for gravity compensation 
# effector_pid      = robotcontrollers.EndEffectorPID( model )
effector_pid      = EndEffectorGravityPID( model )

effector_pid.rbar = r_desired
effector_pid.kp   = np.array([ 10,  10 ])
effector_pid.kd   = np.array([  0,  0 ])

# Closed-loops

robot_with_effector_pid = effector_pid + torque_controlled_robot 

# Simulations
tf = 4
robot_with_effector_pid.x0 = np.array([0,0,0,0])
robot_with_effector_pid.compute_trajectory( tf )
# robot_with_effector_pid.plot_trajectory('xu')
robot_with_effector_pid.animate_simulation()
