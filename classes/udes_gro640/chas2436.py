#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
@author: Simon Chamorro CHAS2436
------------------------------------

Problématique GRO640


"""

import numpy as np

from pyro.control.robotcontrollers import RobotController
from pyro.control.robotcontrollers import EndEffectorPID
from pyro.control.robotcontrollers import EndEffectorKinematicController



###################
# Part 1
###################



def dh2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """
    
    T = np.zeros((4,4))

    c_t = np.cos(theta)
    s_t = np.sin(theta)
    c_a = np.cos(alpha)
    s_a = np.sin(alpha)

    # Transform matrix from DH params
    T = np.array([[c_t, -s_t*c_a, s_t*s_a,  r*c_t], \
                  [s_t, c_t*c_a,  -c_t*s_a, r*s_t], \
                  [0,   s_a,      c_a,      d], \
                  [0,   0,        0,        1]])
    
    return T



def dhs2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    
    WTT = np.zeros((4,4))

    # Check lenght of arrays is equal
    assert len(r) == len(d) == len(theta) == len(alpha)

    # Compute transform matrix
    for i in range(len(r)):
        if i == 0:
            WTT = dh2T( r[i] , d[i], theta[i], alpha[i] )
        else:
            WTT = WTT.dot(dh2T( r[i] , d[i] , theta[i], alpha[i] ))
    
    return WTT



def f(q):
    """

    Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    x : float 3x1 
        Effector (x,y,z) position

    """
    x = np.zeros((3,1))
    
    # Robot DH Parameters
    d     = np.array([0.072, 0.075,        0,              0,     0,              0.217,   q[5]])
    theta = np.array([0,     np.pi + q[0], np.pi/2 + q[1], q[2],  np.pi/2 + q[3], q[4],    0   ])
    r     = np.array([0,     0.033,        0.155,          0.136, 0,              0,       0   ])
    alpha = np.array([0,     np.pi/2,      0,              0,     np.pi/2,       -np.pi/2, 0   ])

    # Compute transform matrix
    T = dhs2T( r , d , theta, alpha )

    # End effector position
    x = np.array([ T[0][3], T[1][3], T[2][3] ])
    
    return x



###################
# Part 2
###################
    


class CustomPositionController( EndEffectorKinematicController ) :
    
    def __init__(self, manipulator ):
        """ """
        
        super().__init__( manipulator)
        self.speed_gain = 1
    
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual

        # Effector space speed
        dr_r = e * self.gains
        
        # Transpose Jacobian
        J_t = np.transpose(J)

        # Position control with speed regulation
        term_1 = np.linalg.inv( np.dot(J_t, J) + self.speed_gain**2 * np.identity(3) )
        term_2 = np.dot( J_t, dr_r)
        dq = np.dot( term_1, term_2 ) 
        
        return dq
    

    
###################
# Part 3
###################
        

        
class CustomDrillingController( EndEffectorPID ) :

    def __init__(self, robot_model ):
        
        EndEffectorPID.__init__( self , robot_model )
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        self.robot = robot_model
        self.pos_k = 400
        self.pos_d = 200
        self.done = False
        self.should_drill = False
        self.hybrid = True
        
        # Target effector force
        self.r_desired = np.array([0.25, 0.25, 0.4]) 
        self.rbar = np.array([0.0, 0.0, -200.0]) 
        

    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        r_actual = self.fwd_kin(q)

        # Jacobian
        J = self.J(q)
        Jt = J.T
        dr = np.dot(J, dq)

        # Position error 
        r_e = self.r_desired - r_actual
        k_e = self.pos_k * np.identity(3)
        b_e = self.pos_d * np.identity(3)

        if self.done:
            f_e = np.zeros(3)

        elif self.should_drill:
            # Force controller
            # Force = ref
            f_e = r
            self.done = r_actual[2] <= 0.2

        else:
            # Position controller
            f_e = np.dot(k_e, r_e) - np.dot(b_e, dr)
            dist_hole = np.sqrt(r_e[0]**2 + r_e[1]**2)
            self.should_drill = dist_hole < 0.001
        
        tau = np.dot( J.T , f_e )
        tau += self.robot.g(q)
        
        return tau
        

    
###################
# Part 4
###################
        

    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return r, dr, ddr


def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return q, dq, ddq



def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return tau

