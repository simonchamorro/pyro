# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:41:26 2018

@author: Alexandre
"""
###############################################################################
import numpy as np

###############################################################################
from pyro.analysis import Trajectory
from pyro.control  import controller
from pyro.signal   import timefiltering



###############################################################################
class OpenLoopController( controller.StaticController ) :
    """  Open-loop controller based on trajectory solution  """
    ############################
    def __init__(self, trajectory   ):
        """ """
        
        # Sys
        self.trajectory = trajectory
        
        # Dimensions
        self.k = 1   
        self.m = trajectory.m
        self.n = trajectory.n
        self.p = trajectory.n
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Open Loop Controller'

    #############################
    def c( self , y , r , t  ):
        """  U depends only on time """
        
        u = self.trajectory.t2u( t )
        
        return u

    @staticmethod
    def load_from_file(name):
        traj = Trajectory.load(name)
        return OpenLoopController(traj)
