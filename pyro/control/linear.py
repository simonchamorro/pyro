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
    r  : reference signal vector  k x 1
    y  : sensor signal vector     k x 1
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
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = np.zeros(self.m) # State derivative vector
        
        e = r - y
        u = e * self.gain
        
        return u
    

class PIDController(controller.StatefulController):
    """General n-state PID controller

    Parameters
    ----------
    KP : array_like
        *m x p* Matrix of proportional controller gain
    KI : array_like
        *m x p* Matrix of integral controller gain
    KD : array_like
        *m x p* Matrix of derivative controller gain
    dv_tau : float, optional
        Time constant of derivative filter.

    Attributes
    ----------
    n : int
        number of controller states
    m : int
        number of controller outputs
    p : int
        number of controller inputs

    Notes
    -----
    The error derivative is filtered and computed according to governing equations
    from:
    https://www.mathworks.com/help/physmod/sps/ref/filteredderivativediscreteorcontinuous.html

    """

    def __init__(self, KP, KI, KD, dv_tau=3E-3):
        self.KI = KI
        self.KP = KP
        self.KD = KD

        self.dv_tau = dv_tau

        self.m = self.KP.shape[0]
        self.p = self.KP.shape[1]
        self.n = self.p * 2

    def f(self, x_ctl, y, r):
        """Evaluate derivative of controller state"""

        if x_ctl.shape != (self.n,):
            return ValueError("Expected x_ctl with shape (%d,)" % self.n)
        if y.shape != (self.p,) or r.shape != (self.p,):
            return ValueError("Expected r and y with shape (%d,)" % self.p)

        # Error
        e = r - y

        # Integrator state derivative
        dx_int = e

        # Filtered derivative state
        x_dv = self.get_x_dv(x_ctl)
        dx_dv = (e - x_dv) / self.dv_tau

        dx = np.stack([dx_int, dx_dv], axis=0)
        assert dx.shape == (self.n,)
        return dx

    def c(self, x_ctl, y, r):
        # Instantaneous error
        e = r - y

        # Error integral value
        I_e = self.get_x_int(x_ctl)

        # Error derivative value
        D_e = (e - self.get_x_dv(x_ctl)) / self.dv_tau

        return self.KP.dot(e) + self.KI.dot(I_e) + self.KD.dot(D_e)

    def get_x_int(self, x_ctl):
        return x_ctl[:self.p]

    def get_x_dv(self, x_ctl):
        return x_ctl[self.p:]

    def get_initial_state(self, sys, x0_sys):
        """Evaluate the initial condition for the numerical solution"""
        error = r - y


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    pass