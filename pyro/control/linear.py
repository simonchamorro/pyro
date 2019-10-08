# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:37:48 2018

@author: alxgr
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.control import controller
###############################################################################



###############################################################################
# Simple proportionnal controller
###############################################################################
        
class ProportionnalSingleVariableController( controller.StaticController ) :
    """ 
    Simple proportionnal compensator
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     k x 1
    u  : control inputs vector    k x 1
    t  : time                     1 x 1
    ---------------------------------------
    u = c( y , r , t ) = (r - y) * gain

    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k = 1):
        """ """
        
        # Dimensions
        self.k = k   
        self.m = k   
        self.p = k
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Proportionnal Controller'
        
        # Gains
        self.gain = 1
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
        r  : reference signal_proc vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) # State derivative vector
        
        e = r - y
        u = e * self.gain
        
        return u
    




'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    pass