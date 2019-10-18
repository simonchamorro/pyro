# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np

from scipy.integrate import odeint

from .graphical import TrajectoryPlotter

##########################################################################
# Simulation Objects
##########################################################################

class Trajectory():
    """Simulation data"""

    _dict_keys = ['x', 'u', 't', 'dx', 'y', 'r', 'J', 'dJ']

    def __init__(self, x, u, t, dx, y, r=None, J=None, dJ=None):
        """
        x:  array of dim = ( time-steps , sys.n )
        u:  array of dim = ( time-steps , sys.m )
        t:  array of dim = ( time-steps , 1 )
        y:  array of dim = ( time-steps , sys.p )
        """

        self.x  = x
        self.u  = u
        self.t  = t
        self.dx = dx
        self.y  = y
        self.r  = r
        self.J  = J
        self.dJ = dJ

        self._compute_size()
        
    ############################
    def _asdict(self):
        return {k: getattr(self, k) for k in self._dict_keys}
    
    ############################
    def save(self, name = 'trajectory_solution.npy' ):
        np.savez(name , **self._asdict())
    
    ############################
    @classmethod
    def load(cls, name):
        try:
            # try to load as new format (np.savez)
            with np.load(name) as data:
                return cls(**data)

        except ValueError:
            # If that fails, try to load as "legacy" numpy object array
            data = np.load(name, allow_pickle=True)
            return cls(*data)
        
    ############################
    def _compute_size(self):
        self.time_final = self.t.max()
        self.time_steps = self.t.size

        self.n = self.x.shape[1]
        self.m = self.u.shape[1]

        # Check consistency between signals
        for arr in [self.x, self.y, self.u, self.dx, self.r, self.J, self.dJ]:
            if (arr is not None) and (arr.shape[0] != self.time_steps):
                raise ValueError("Result arrays must have same length along axis 0")
                

    ############################
    def t2u(self, t ):
        """ get u from time """

        if t > self.time_final:
            raise ValueError("Got time t greater than final time")
        
        # Find time index
        i = (np.abs(self.t - t)).argmin()

        # Find associated control input
        u = self.u[i,:]

        return u

    ############################
    def t2x(self, t ):
        """ get x from time """

        # Find time index
        i = (np.abs(self.t - t)).argmin()

        # Find associated control input
        return self.x[i,:]


class Simulator:
    """Simulation Class for open-loop ContinuousDynamicalSystem

    Parameters
    -----------
    cds    : Instance of ContinuousDynamicSystem
    tf     : float : final time for simulation
    n      : int   : number of time steps
    solver : {'ode', 'euler'}
    """
    
    ############################
    def __init__(
        self, ContinuousDynamicSystem, tf=10, n=10001, solver='ode'):

        self.cds    = ContinuousDynamicSystem
        self.t0     = 0
        self.tf     = tf
        self.n      = int(n)
        self.dt     = ( tf + 0.0 - self.t0 ) / ( n - 1 )
        self.solver = solver
        self.x0     = self.cds.x0
        
        # Check Initial condition state-vector
        if self.x0.size != self.cds.n:
            raise ValueError(
                "Number of elements in x0 must be equal to number of states"
            )
            
        self.traj = None

    ##############################
    def compute(self):
        """ Integrate trought time """

        t  = np.linspace( self.t0 , self.tf , self.n )

        if self.solver == 'ode':

            x_sol = odeint( self.cds.fsim , self.x0 , t)

            # Compute inputs-output values
            y_sol  = np.zeros(( self.n , self.cds.p ))
            u_sol  = np.zeros((self.n,self.cds.m))
            dx_sol = np.zeros((self.n,self.cds.n))

            for i in range(self.n):
                ti = t[i]
                xi = x_sol[i,:]
                ui = self.cds.t2u( ti )

                dx_sol[i,:] = self.cds.f( xi , ui , ti )
                y_sol[i,:]  = self.cds.h( xi , ui , ti )
                u_sol[i,:]  = ui

        elif self.solver == 'euler':

            x_sol  = np.zeros((self.n,self.cds.n))
            dx_sol = np.zeros((self.n,self.cds.n))
            u_sol  = np.zeros((self.n,self.cds.m))
            y_sol  = np.zeros((self.n,self.cds.p))

            # Initial State
            x_sol[0,:] = self.x0
            dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
            for i in range(self.n):

                ti = t[i]
                xi = x_sol[i,:]
                ui = self.cds.t2u( ti )

                if i+1<self.n:
                    dx_sol[i]    = self.cds.f( xi , ui , ti )
                    x_sol[i+1,:] = dx_sol[i] * dt + xi

                y_sol[i,:] = self.cds.h( xi , ui , ti )
                u_sol[i,:] = ui

        traj = Trajectory(
          x = x_sol,
          u = u_sol,
          t = t,
          dx= dx_sol,
          y = y_sol
        )
        
        self.traj = traj
        
        return traj


###############################################################################
# Closed Loop Simulation
###############################################################################
    
class CLosedLoopSimulator(Simulator):
    """ 
    Simulation Class for closed-loop ContinuousDynamicalSystem 
    --------------------------------------------------------
    CLSystem  : Instance of ClosedLoopSystem
    tf : final time
    n  : number if point
    solver : 'ode' or 'euler'
    --------------------------------------------------------
    Use this class instead of Simulation() in order to access
    internal control inputs
    """
    
    ###########################################################################
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sys = self.cds.cds
        self.ctl = self.cds.ctl

    ###########################################################################
    def compute(self):

        sol = super().compute()
        r, u = self._compute_inputs(sol)

        cltraj = Trajectory(
            x  = sol.x,
            u  = u,
            t  = sol.t,
            dx = sol.dx,
            y  = sol.y,
            r  = r
        )

        return cltraj

    ###########################################################################
    def _compute_inputs(self, sol):
        """ Compute internal control signal_proc of the closed-loop system """

        r_sol = sol.u.copy() # reference is input of combined sys
        u_sol = np.zeros((self.n,self.sys.m))

        # Compute internal input signal_proc
        for i in range(self.n):

            ri = r_sol[i,:]
            yi = sol.y[i,:]
            ti = sol.t[i]

            u_sol[i,:] = self.ctl.c( yi , ri , ti )

        return (r_sol, u_sol)
