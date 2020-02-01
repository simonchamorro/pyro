# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:40:31 2018

@author: alxgr
"""

from abc import ABC, abstractmethod

from copy import copy

import numpy as np

from pyro.dynamic import system

from pyro.analysis import phaseanalysis
from pyro.analysis import simulation
from pyro.analysis import costfunction

###############################################################################
# Mother Controller class
###############################################################################

class StaticController():
    """ 
    Mother class for memoryless controllers
    ---------------------------------------
    r  : reference signal vector       k x 1
    y  : sensor signal vector          p x 1
    u  : control inputs vector         m x 1
    t  : time                          1 x 1
    ---------------------------------------
    u = c( y , r , t )
    
    """
    
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    
    ############################
    def __init__(self, k=1, m=1, p=1):
        """ """
        # System parameters to be implemented
        
        # Dimensions
        self.k = k   
        self.m = m   
        self.p = p
        
        # Label
        self.name = 'StaticController'
        
        # Reference signal_proc info
        self.ref_label = []
        self.ref_units = []
        
        for i in range(k):
            self.ref_label.append('Ref. '+str(i))
            self.ref_units.append('')
        
        self.r_ub = np.zeros(self.k) + 10 # upper bounds
        self.r_lb = np.zeros(self.k) - 10 # lower bounds
        
        # default constant reference
        self.rbar = np.zeros(self.k)
        
    
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
        
        raise NotImplementedError
        
        return u
    
    #########################################################################
    # Default methods that can be overloaded in child classes
    #########################################################################
    
    #############################
    def t2r( self , t ):
        """ 
        Reference signal fonction u = t2u(t)
        
        INPUTS
        t  : time                     1 x 1
        
        OUTPUTS
        r  : controller reference vector    m x 1
        
        Defaul is a constant signal equal to self.rbar, can overload the
        with a more complexe reference signal time-function 
        
        """
        
        #Default is a constant signal
        r = self.rbar
        
        return r
    
    
    #########################################################################
    # No need to overwrite the following functions for child classes
    #########################################################################
    
    #############################
    def cbar( self , y , t = 0 ):
        """ 
        Feedback static computation u = c( y, r = rbar, t) for
        default reference
        
        INPUTS
        y  : sensor signal vector     p x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        u = self.c( y , self.rbar , t )
        
        return u
    
    
    #############################
    def __add__(self, sys):
        """ 
        closed_loop_system = controller + dynamic_system
        """
        
        cl_sys = ClosedLoopSystem( sys , self )
        
        return cl_sys
    
    

###############################################################################
# Mother "Static controller + dynamic system" class
###############################################################################

class ClosedLoopSystem( system.ContinuousDynamicSystem ):
    """ 
    Dynamic system connected with a static controller
    ---------------------------------------------
    NOTE: 
    Ignore any feedthough in the plant to avoid creating algebraic loops
    This is only valid if the output function h is not a fonction of u
    New equations assume y = h(x,u,t) -- > y = h(x,t)

    """
    ############################
    def __init__(self, ContinuousDynamicSystem , StaticController ):
        """ """
        
        self.plant      = ContinuousDynamicSystem
        self.controller = StaticController
        
        ######################################################################
        # Check dimensions match
        if not (self.plant.m == self.controller.m ):
            raise NameError('Dimension mismatch between controller and' + 
            ' dynamic system for the input signal u')
        elif not (self.plant.p == self.controller.p ):
            raise NameError('Dimension mismatch between controller and' + 
            ' dynamic system for the output signal y')
        ######################################################################
        
        # Dimensions of global closed-loop dynamic system
        self.n = self.plant.n
        self.m = self.controller.k 
        self.p = self.plant.p
        
        # Labels
        self.name = ('Closed-Loop ' + self.plant.name + 
                     ' with ' + self.controller.name )
        self.state_label  = self.plant.state_label
        self.input_label  = self.controller.ref_label
        self.output_label = self.plant.output_label
        
        # Units
        self.state_units  = self.plant.state_units
        self.input_units  = self.controller.ref_units
        self.output_units = self.plant.output_units
        
        # Define the domain
        self.x_ub = self.plant.x_ub
        self.x_lb = self.plant.x_lb
        self.u_ub = self.controller.r_ub
        self.u_lb = self.controller.r_lb
        
        # Default State and inputs        
        self.xbar = self.plant.xbar
        self.ubar = self.controller.rbar
        self.ybar = self.plant.ybar
        
        ################################
        # Variables
        ################################
        
        # Initial value for simulations
        self.x0   = self.plant.x0
        
        # Result of last simulation
        self.traj = None
        
        # Cost function for evaluation
        # default is a quadratic cost function with diag Q and R matrices
        self.cost_function = costfunction.QuadraticCostFunction.from_sys(self)
        
    
    ###########################################################################
    def f( self , x , u , t ):
        """ 
        Continuous time foward dynamics evaluation dx = f(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vector  n x 1
        
        """
        
        dx = np.zeros(self.n) # State derivative vector
        
        r = u # input of closed-loop global sys is ref of the controller
        
        # Compute output signal
        y = self.plant.h( x, self.plant.ubar, t)
        
        # Compute control inputs
        u = self.controller.c( y, r, t)
        
        # Compute state derivatives
        dx = self.plant.f( x, u, t)
        
        return dx
    

    ###########################################################################
    def h( self , x , u , t ):
        """ 
        Output fonction y = h(x,u,t)
        
        INPUTS
        x  : state vector             n x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        
        OUTPUTS
        y  : output derivative vector o x 1
        
        """
        
        #y = np.zeros(self.p) # Output vector
        
        y = self.plant.h( x , self.plant.ubar , t )
        
        return y
    
    
    ###########################################################################
    def t2u( self , t ):
        """ 
        Reference signal fonction u = t2u(t)
        
        INPUTS
        t  : time                     1 x 1
        
        OUTPUTS
        u  : control inputs vector    m x 1
        
        Defaul is a constant signal equal to self.ubar, can overload the
        with a more complexe reference signal time-function 
        
        """
        
        # Input of closed-loop global sys is ref of the controller
        u = self.controller.t2r(t)
        
        return u
    
    
    ###########################################################################
    def plot_phase_plane_closed_loop(self , x_axis = 0 , y_axis = 1 ):
        """ 
        Plot Phase Plane vector field of the system
        ------------------------------------------------
        
        x_axis : index of state on x axis
        y_axis : index of state on y axis
        
        """

        pp = phaseanalysis.PhasePlot( self , x_axis , y_axis )
        
        pp.compute_grid()
        pp.plot_init()
        
        # Closed-loop Behavior
        pp.color = 'r'
        pp.compute_vector_field()
        pp.plot_vector_field()
        
        # Open-Loop Behavior
        pp.f     = self.plant.f
        pp.ubar  = self.plant.ubar
        pp.color = 'b'
        pp.compute_vector_field()
        pp.plot_vector_field()
        
        pp.plot_finish()
        
        return pp
        
    
    #############################
    def compute_trajectory(
        self, tf=10, n=10001, solver='ode'):
        """ 
        Simulation of time evolution of the system
        ------------------------------------------------
        tf : final time
        n  : time steps
        """

        sim = simulation.CLosedLoopSimulator(self, tf, n, solver)
        
        self.traj = sim.compute()

        return self.traj


    #############################################
    # Make graph function use the internal sys
    #############################################

    def get_plotter(self):
        return self.plant.get_plotter()

    def get_animator(self):
        return self.plant.get_animator()
    

    ###########################################################################
    def show(self, q , x_axis = 0 , y_axis = 1 ):
        """ Plot figure of configuration q """
        
        system.ContinuousDynamicSystem.show( self.plant , q , 
                                            x_axis = 0 , y_axis = 1  )
        
    
    ###########################################################################
    def show3(self, q ):
        """ Plot figure of configuration q """
        
        system.ContinuousDynamicSystem.show3(self.plant, q)
        
        
    ###########################################################################
    def plot_phase_plane_trajectory_closed_loop(self, traj, x_axis=0, y_axis=1):
        self.get_plotter().phase_plane_trajectory_closed_loop(traj, x_axis, y_axis)




###############################################################################
class StatefulController(ABC):
    """
    
    Controller with internal states (memory)
    
    ex: integral action of a PID

    Parameters
    ----------
    n : int
        number of controller states
    m : int
        number of controller outputs
    p : int
        number of controller inputs

    """
    def __init__(self, n, m, p):
        self.n = n
        self.m = m
        self.p = p
        self.k = p

        self.name = "StatefulController"

        self.ref_label = ["Ref. %d" % i for i in range(self.p)]
        self.ref_units = [''] * self.p

        # Default constant reference signal
        self.rbar = np.zeros(self.p)

        # Default bounds
        self.r_ub = np.zeros(self.k) + 10 # upper bounds
        self.r_lb = np.zeros(self.k) - 10 # lower bounds

    @abstractmethod
    def c(self, xctl, y, r, t):
        """Controller output function"""
        return np.zeros(self.m)

    @abstractmethod
    def f(self, xctl, y, r, t):
        """Differential equation describing the internal controller states.

        In the form ```dx = f(x, y, r, t)``. This equation is numerically solved during
        system simulation.
        """
        return np.zeros(self.n)

    def __add__(self, sys):
        return StatefulCLSystem(sys, self)


##############################################################################
class StatefulCLSystem(ClosedLoopSystem):
    """Closed loop system with stateful controller
    """
    def __init__(self, cds, ctl):
        super().__init__(cds, ctl)

        # Add extra states that represent system memory
        self.n = self.plant.n + self.controller.n

        if cds.p != ctl.p:
            raise ValueError("Controller inputs do not match system outputs")
        if cds.m != ctl.m:
            raise ValueError("Controller outputs do not match system inputs")

    def f(self, x, u, t):
        x_sys, x_ctl = self._split_states(x)

        # Input to CL system interpreted as reference signal
        r = u

        # Eval current system output. Assume there is no feedforward term, as it
        # would cause an algebraic loop
        y = self.plant.h(x_sys, self.plant.ubar, t)

        # input u to dynamic system evaluated by controller
        u = self.controller.c(x_ctl, y, r, t)

        dx_sys = self.plant.f(x_sys, u, t)
        dx_ctl = self.controller.f(x_ctl, y, r, t)

        dx = np.concatenate([dx_sys, dx_ctl], axis=0)
        assert dx.shape == (self.n,)
        return dx

    def h(self, x, u, t):
        x_sys, _ = self._split_states(x)
        return self.plant.h(x_sys, u, t)

    def compute_trajectory(self , x0, r=None, **kwargs):
        """Simulation of time evolution of the system

        Parameters
        ----------
        x0_sys : array_like
            Vector of size (`self.plant.n`) representing the initial state values for the
            dynamic system. Initial values for the controller are calculated internally.

        See `ClosedLoopSystem.compute_trajectory` for description of other paramters.

        """

        if r is None:
            r = lambda t: self.controller.rbar

        x0 = np.asarray(x0).flatten()
        if x0.shape != (self.plant.n,):
            raise ValueError("Expected x0 of shape (%d,)" % self.plant.n)

        x0_ctl = self.controller.get_initial_state(self.plant, x0, r)
        x0_full = np.concatenate([x0, x0_ctl], axis=0)

        return super().compute_trajectory(x0=x0_full, r=r, **kwargs)

    def _compute_inputs(self, sol):
        """ Compute internal control signal of the closed-loop system """

        r_sol = sol.u.copy() # reference is input of combined sys
        u_sol = np.empty((sol.n, self.controller.m))

        # Compute internal input signal
        for i in range(r_sol.shape[0]):

            ri = r_sol[i,:]
            yi = sol.y[i,:]
            ti = sol.t[i]
            _, x_ctl = self._split_states(sol.x[i, :])

            u_sol[i,:] = self.controller.c(x_ctl, yi, ri, ti)

        new_sol = copy(sol)
        new_sol.r = r_sol
        new_sol.u = u_sol

        return new_sol

    def _split_states(self, x):
        """Separate full state vector into system and controller states"""
        x_sys, x_ctl = x[:self.plant.n], x[self.plant.n:]
        assert x_ctl.shape == (self.controller.n,)
        return (x_sys, x_ctl)


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass

    
