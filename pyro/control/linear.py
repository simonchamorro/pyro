# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:37:48 2018

@author: alxgr
"""
###############################################################################
import numpy as np
###############################################################################
from . import controller

from .._utils import to_2D_arr
###############################################################################

class ProportionalController(controller.StaticController):
    """General (SISO or MIMO) proportional controller."""
    def __init__(self, KP):
        self.KP = to_2D_arr(KP)

        k = self.KP.shape[1]
        m = self.KP.shape[0]
        p = self.KP.shape[1]
        super().__init__(k, m, p)

<<<<<<< HEAD
        self.rbar = np.zeros((self.k,))
        self.name = "%d X %d Proportional Contrller" % self.KP.shape

    def c(self, y, r, t=0):
        return self.KP.dot(r - y)


class PIDController(controller.StatefulController):
    """General (SISO or MIMO) PID controller

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

    Notes
    -----
    The error derivative is filtered and computed according to governing equations
    from:
    https://www.mathworks.com/help/physmod/sps/ref/filteredderivativediscreteorcontinuous.html

    """

    def __init__(self, KP, KI=None, KD=None, dv_tau=3E-3):
        self.KP = to_2D_arr(KP)

        if KI is None:
            self.KI = np.zeros(self.KP.shape)
        else:
            self.KI = to_2D_arr(KI)
            if self.KI.shape != self.KP.shape:
                raise ValueError("Shape of KI does not match KP")

        if KD is None:
            self.KD = np.zeros(self.KP.shape)
        else:
            self.KD = to_2D_arr(KD)
            if self.KD.shape != self.KP.shape:
                raise ValueError("Shape of KD does not match KP")

        self.dv_tau = dv_tau
        self.name = "PID Controller"

        super().__init__(n=self.KP.shape[1]*2, m=self.KP.shape[0], p=self.KP.shape[1])

    def f(self, x_ctl, y, r, t):
        """Evaluate derivative of controller state"""

        if x_ctl.shape != (self.n,):
            raise ValueError("Expected x_ctl with shape (%d,)" % self.n)
        if y.shape != (self.p,) or r.shape != (self.p,):
            raise ValueError("Expected r and y with shape (%d,)" % self.p)

        # Error
        e = r - y

        # Integrator state derivative
        dx_int = e

        # Filtered derivative state
        x_dv = self.get_x_dv(x_ctl)
        dx_dv = (e - x_dv) / self.dv_tau

        dx = np.concatenate([dx_int, dx_dv], axis=0)
        assert dx.shape == (self.n,)
        return dx

    def c(self, x_ctl, y, r, t):
        if x_ctl.shape != (self.n, ):
            raise ValueError("expected x_ctl with shape (%d,)" % self.n)

        # Instantaneous error
=======
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
        
>>>>>>> ai-value-iteration
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

    def get_initial_state(self, sys, x0_sys, r):
        """Evaluate the initial condition for the numerical solution"""
        y = sys.h(x0_sys, sys.ubar, t=0)
        error = r(0) - y

        x0_int = np.zeros(self.p)
        x0_deriv = error
        x0 = np.concatenate([x0_int, x0_deriv], axis=0)

        assert x0.shape == (self.n,)
        return x0


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":
    pass