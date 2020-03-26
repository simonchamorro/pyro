#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:15:49 2020

@author: alex
"""

import numpy as np
from pyro.control import nonlinear

##############################################################################
        
class SinglePendulumAdaptativeController( nonlinear.ComputedTorqueController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        nonlinear.ComputedTorqueController.__init__( self , model , traj )
        
        self.name = 'Adaptive controller'
        
        # Params
        self.dt = 0.001
        self.A = np.array([0.2,0.2])
        self.T=np.eye(2)
        #self.A[0] = 2
        #self.A[1] = 5
        self.Kd = 1
        self.lam  = 1   # Sliding surface slope
        self.nab  = 0.1 # Min convergence rate
        
        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = self.Kd*s
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
                        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        

        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        
        Y = np.zeros(2)
        dA = np.zeros(2)

        Y[0]=ddq_r
        Y[1]=np.sin(q)
        
        b = Y * s
        dA=-1*np.dot( self.T , b )
        
        self.A=self.A+dA*self.dt  #TODO use dynamic controller class
        #print(self.A)
                
        u                     = self.adaptative_torque( Y , s  , q , t )
        
        return u



##############################################################################
        
class DoublePendulumAdaptativeController( nonlinear.ComputedTorqueController ):
    """ 
    
    """
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        nonlinear.ComputedTorqueController.__init__( self , model , traj )
        
        self.name = 'Adaptive controller'
        
        # Params
        self.dt = 0.001
        self.A = np.zeros(5)
        self.T=np.eye(5)
        self.Kd = np.eye(2)
        self.lam  = 1   # Sliding surface slope
        
        
        
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin usefull in other computation 
        ------------------------------------------------
        
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        
        return [c1,s1,c2,s2,c12,s12]

        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = np.dot(self.Kd,s)
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
                        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        

        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        Y = np.zeros((2,5))
        dA = np.zeros(5)
                
        Y[0,0]=ddq_r[0]*c2
        Y[0,1]=ddq_r[1]*c2
        Y[0,2]=s2*dq[1]*dq_r[0]
        Y[0,3]=s2*(dq[0]+dq[1])*dq_r[1]
        Y[0,4]=s1+s12
        Y[1,0]=ddq_r[0]*c2
        Y[1,1]=ddq_r[1]
        Y[1,2]=s2*dq[0]*dq_r[0]
        Y[1,3]=0
        Y[1,4]=s12
        
        b = np.dot(Y.T,s)
        dA=-1*np.dot( self.T , b )
        
        self.A=self.A+dA*self.dt # ToDo use Dynamic COntroller Class
        #print(self.A)
                
        u                     = self.adaptative_torque( Y , s  , q , t )
        
        return u
    
##############################################################################

