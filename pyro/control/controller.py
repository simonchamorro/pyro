# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:40:31 2018

@author: alxgr
"""

from copy import copy

import numpy as np

from pyro.dynamic import system
from pyro.analysis import phaseanalysis
from pyro.analysis import simulation
from pyro.analysis import graphical
from pyro.analysis import Trajectory

###############################################################################
# Mother Controller class
###############################################################################

class StaticController:
    """ 
    Mother class for memoryless controllers
    ---------------------------------------
    r  : reference signal_proc vector  k x 1
    y  : sensor signal_proc vector     p x 1
    u  : control inputs vector    m x 1
    t  : time                     1 x 1
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
    # No need to overwrite the following functions for child classes
    #########################################################################
    
    #############################
    def cbar( self , y , t = 0 ):
        """ 
        Feedback static computation u = c( y, r = rbar, t) for
        default reference
        
        INPUTS
        y  : sensor signal_proc vector     p x 1
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
    Ignore any feedthough to avoid creating algebraic loop
    This is only valid if the output function h is not a fonction of u
    New equations assume y = h(x,u,t) -- > y = h(x,t)

    """
    ############################
    def __init__(self, ContinuousDynamicSystem , StaticController):
        """ """
        
        self.cds = ContinuousDynamicSystem
        self.ctl = StaticController
        
        ######################################################################
        # Check dimensions match
        if not (self.cds.m == self.ctl.m ):
            raise NameError('Dimension mismatch between controller and' + 
            ' dynamic system for the input signal_proc u')
        elif not (self.cds.p == self.ctl.p ):
            raise NameError('Dimension mismatch between controller and' + 
            ' dynamic system for the output signal_proc y')
        ######################################################################
        
        # Dimensions of global closed-loop dynamic system
        self.n = self.cds.n
        self.m = self.ctl.k 
        self.p = self.cds.p
        
        # Labels
        self.name = 'Closed-Loop ' + self.cds.name + ' with ' + self.ctl.name
        self.state_label  = self.cds.state_label
        self.input_label  = self.ctl.ref_label
        self.output_label = self.cds.output_label
        
        # Units
        self.state_units = self.cds.state_units
        self.input_units = self.ctl.ref_units
        self.output_units = self.cds.output_units
        
        # Define the domain
        self.x_ub = self.cds.x_ub
        self.x_lb = self.cds.x_lb
        self.u_ub = self.ctl.r_ub
        self.u_lb = self.ctl.r_lb
        
        # Default State and inputs        
        self.xbar = self.cds.xbar
        self.ubar = self.ctl.rbar
        
    
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
        y = self.cds.h( x, self.cds.ubar, t)
        u = self.ctl.c( y, r, t)
        
        dx = self.cds.f( x, u, t)
        
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
        
        y = self.cds.h( x , self.cds.ubar , t )
        
        return y
    
    ###########################################################################
    def plot_phase_plane_closed_loop(self , x_axis = 0 , y_axis = 1 ):
        """ 
        Plot Phase Plane vector field of the system
        ------------------------------------------------
        
        x_axis : index of state on x axis
        y_axis : index of state on y axis
        
        """

        self.pp = phaseanalysis.PhasePlot( self , x_axis , y_axis )
        
        self.pp.compute_grid()
        self.pp.plot_init()
        
        # Closed-loop Behavior
        self.pp.color = 'r'
        self.pp.compute_vector_field()
        self.pp.plot_vector_field()
        
        # Open-Loop Behavior
        self.pp.f     = self.cds.f
        self.pp.ubar  = self.cds.ubar
        self.pp.color = 'b'
        self.pp.compute_vector_field()
        self.pp.plot_vector_field()
        
        self.pp.plot_finish()
        
    
    ###########################################################################
    def compute_trajectory(self , x0, tf=10 , n=10001 , solver='ode', costfunc=None):
        """ 
        Simulation of time evolution of the system
        ------------------------------------------------
        x0 : initial time
        tf : final time
        
        """

        sol = simulation.Simulator(self, tf, n, solver, x0=x0).compute()
        sol = self._compute_inputs(sol)

        if costfunc is not None:
            sol = costfunc.eval(sol)

        return sol

    def _compute_inputs(self, sol):
        """ Compute internal control signal_proc of the closed-loop system """

        r_sol = sol.u.copy() # reference is input of combined sys
        u_sol = np.empty((sol.n, self.ctl.m))

        # Compute internal input signal_proc
        for i in range(r_sol.shape[0]):

            ri = r_sol[i,:]
            yi = sol.y[i,:]
            ti = sol.t[i]

            u_sol[i,:] = self.ctl.c( yi , ri , ti )

        new_sol = copy(sol)
        new_sol.r = r_sol
        new_sol.u = u_sol

        return new_sol


    #############################################
    # Make graph function use the internal sys
    #############################################

    def get_plotter(self):
        return self.cds.get_plotter()

    def get_animator(self):
        return self.cds.get_animator()

    ###########################################################################
    def show(self, q , x_axis = 0 , y_axis = 1 ):
        """ Plot figure of configuration q """
        
        system.ContinuousDynamicSystem.show( self.cds , q , 
                                            x_axis = 0 , y_axis = 1  )
        
    
    ###########################################################################
    def show3(self, q ):
        """ Plot figure of configuration q """
        
        system.ContinuousDynamicSystem.show3(self.cds, q)


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass

    
