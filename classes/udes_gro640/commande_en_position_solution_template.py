#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""
import numpy as np

from gro640_robots import Robot1, Robot3

from pyro.control.robotcontrollers import EndEffectorKinematicController


class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        """ """
        
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        
        #################

        # Vos paramètres de loi de commande ici !!
        
        
        
        
        
        
        
        #################
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
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
        
        ##################
        # TODO
        
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        # Votre loi de commande ici !!!
        
        
        
        
        
        
        
        
        
        
        
        ####################
        
        return dq


# Model cinématique du robot
sys = Robot1()

# Contrôleur en position de l'effecteur standard
ctl = CustomPositionController( sys )

# Cible de position pour l'effecteur
ctl.rbar = np.array([0,-1])

# Dynamique en boucle fermée
clsys = ctl + sys

# Configurations de départs

# clsys.x0 =  np.array([0,0.1,0])  #crash
# clsys.x0 =  np.array([0,0.3,0]) #crash 
# clsys.x0 =  np.array([0,0.5,0]) #crash 
clsys.x0 =  np.array([0,0.7,0]) # fonctionne

# Simulation
clsys.compute_trajectory( 5 )
clsys.plot_trajectory( plot='xu' )
clsys.animate_simulation()