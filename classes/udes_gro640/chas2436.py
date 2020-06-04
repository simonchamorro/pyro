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
import matplotlib.pyplot as plt

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
    theta = np.array([0,     np.pi + q[0], np.pi/2 - q[1], -q[2], np.pi/2 - q[3], -q[4],   0   ])
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
        self.speed_gain = 0.5
    
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
        dist_hole = np.sqrt(r_e[0]**2 + r_e[1]**2)

        if r_actual[2] <= 0.2:
            f_e = np.zeros(3)

        elif dist_hole < 0.01:
            # Force controller
            # Force = ref
            if self.hybrid:
                k_e[2][2] = 0
                b_e[2][2] = 0
                f_e = r + np.dot(k_e, r_e) - np.dot(b_e, dr)
            else:
                f_e = r

        else:
            # Position controller
            f_e = np.dot(k_e, r_e) - np.dot(b_e, dr)
        
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
    l = 30 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    # 5th polynomial 1d trajectory parameters
    fifth_poly_matrix = np.array([[0,         0,         0,        0,      0,   1], \
                                  [t_f**5,    t_f**4,    t_f**3,   t_f**2, t_f, 1], \
                                  [0,         0,         0,        0,      1,   0], \
                                  [5*t_f**4,  4*t_f**3,  3*t_f**2, 2*t_f,  1,   0], \
                                  [0,         0,         0,        2,      0,   0], \
                                  [20*t_f**3, 12*t_f**2, 6*t_f,    2,      0,   0]])
    
    # Initial and terminal conditions
    s0_vec = np.array([0, 1, 0, 0, 0, 0])

    # Find polynomial coefficients
    poly_matrix_inv = np.linalg.inv(fifth_poly_matrix)
    coeff = np.dot(poly_matrix_inv, s0_vec)

    time = np.linspace(0, t_f, num=l, endpoint=True)
    dist = np.linalg.norm(r_f - r_0)

    # Compute traj, speed and accel
    for idx, t in enumerate(time):
        s_t = np.dot(coeff, np.array([t**5, t**4, t**3, t**2, t, 1]))
        ds_t = np.dot(coeff, np.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0]))
        dds_t = np.dot(coeff, np.array([20*t**3, 12*t**2, 6*t, 2, 0, 0]))

        r[:,idx] = r_0 + s_t*(r_f - r_0)
        dr[:,idx] = ds_t*(r_f - r_0)
        ddr[:,idx] = dds_t*(r_f - r_0)

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
    
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3
    
    for i in range(l):
        # Find q     
        c3 = ((r[0,i]**2) + (r[1,i]**2) + ((r[2,i]- l1)**2) - (l2**2) - (l3**2)) / (2*l2*l3)                 
        s3 = np.sqrt(1-(c3**2))           
        q[0,i] = np.arctan2(r[1,i],r[0,i])         
        q[1,i] = np.arctan2( (r[2,i] - l1), (np.sqrt(r[0,i]**2 + r[1,i]**2)) ) \
               - np.arctan2(l3*s3, l2 + l3 * c3)         
        q[2,i] = np.arctan2(s3,c3)

        # Find dq
        J = manipulator.J(q[:,i])
        J_inv = np.linalg.inv(J)
        dq[:,i] = np.dot(J_inv, dr[:,i])

        # Compute dJ
        dJq1 = np.array([[-np.cos(q[0,i])*(l2*np.cos(q[1,i])+l3*np.cos(q[1,i]+q[2,i])),         np.sin(q[0,i])*(l2*np.sin(q[1,i])+l3*np.sin(q[1,i]+q[2,i])),          l3*np.sin(q[0,i])*np.sin(q[1,i]+q[2,i])], \
                         [-np.sin(q[0,i])*(l2*np.cos(q[1,i])+l3*np.cos(q[1,i]+q[2,i])),         -np.cos(q[0,i])*(l2*np.sin(q[1,i])+l3*np.sin(q[1,i]+q[2,i])),         -l3*np.cos(q[0,i])*np.sin(q[1,i]+q[2,i])], \
                         [0,                                                                    0,                                                                    0]])

        dJq2 = np.array([[np.sin(q[0,i])*(l2*np.sin(q[1,i])+l3*np.sin(q[1,i]+q[2,i])*dq[1,i]),  -np.cos(q[0,i])*(l2*np.cos(q[1,i])+l3*np.cos(q[1,i]+q[2,i])*dq[1,i]), -l3*np.cos(q[0,i])*np.cos(q[1,i]+q[2,i])], \
                         [-np.cos(q[0,i])*(l2*np.sin(q[1,i])+l3*np.sin(q[1,i]+q[2,i])),         -np.sin(q[0,i])*(l2*np.cos(q[1,i])+l3*np.cos(q[1,i]+q[2,i])),         -l3*np.sin(q[0,i])*np.cos(q[1,i]+q[2,i])], \
                         [0,                                                                    l2*np.sin(q[1,i])*dq[1,i] + l3*np.sin(q[1,i]+q[2,i])*dq[1,i],         l3*np.sin(q[1,i]+q[2,i])*dq[1,i]]])

        dJq3 = np.array([[+l3*np.sin(q[0,i])*np.sin(q[1,i]+q[2,i]),                             -l3*np.cos(q[0,i])*np.cos(q[1,i]+q[2,i]),                             -l3*np.cos(q[0,i])*np.cos(q[1,i]+q[2,i])], \
                         [-l3*np.cos(q[0,i])*np.sin(q[1,i]+q[2,i]),                             -l3*np.sin(q[0,i])*np.cos(q[1,i]+q[2,i]),                             -l3*np.sin(q[0,i])*np.cos(q[1,i]+q[2,i])], \
                         [0,                                                                    l3*np.sin(q[1,i]+q[2,i])*dq[2,i],                                     l3*np.sin(q[1,i]+q[2,i])*dq[2,i]]])

        # Find ddq
        dJ = dq[0,i]*dJq1 + dq[1,i]*dJq2 + dq[2,i]*dJq3
        ddq[:,i] = np.dot(J_inv, ddr[:,i] - np.dot(dJ, dq[:,i]))

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
    
    for i in range(l):
        # Get robot matrices
        H = manipulator.H(q[:,i])
        C = manipulator.C(q[:,i], dq[:,i])
        D = manipulator.d(q[:,i], dq[:,i])
        B_inv = np.linalg.inv(manipulator.B(q[:,i]))

        # Calculate torque for desired accel
        tau[:,i] = np.dot(B_inv, np.dot(H, ddq[:,i]) \
                               + np.dot(C, dq[:,i]) \
                               + np.dot(D, dq[:,i]) \
                               + manipulator.g(q[:,i]))
   
    return tau

