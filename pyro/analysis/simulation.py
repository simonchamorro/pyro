# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import odeint

from .graphical import TrajectoryPlotter

##################################################################### #####
# Simulation Objects
##########################################################################

class Trajectory() :
    """Simulation data"""

    _dict_keys = ('x_sol', 'u_sol', 't', 'dx_sol', 'y_sol')

    ############################
    def __init__(self, x, u, t, dx, y):
        """
        x:  array of dim = ( time-steps , sys.n )
        u:  array of dim = ( time-steps , sys.m )
        t:  array of dim = ( time-steps , 1 )
        y:  array of dim = ( time-steps , sys.p )
        """

        self.x_sol  = x
        self.u_sol  = u
        self.t  = t
        self.dx_sol = dx
        self.y_sol  = y

        self._compute_size()

    ############################
    def _to_dict(self):
        return {k: getattr(self, k) for k in self._dict_keys}

    def save(self, name = 'trajectory_solution.npy' ):
        as_dict = self._to_dict()
        np.savez( name , **as_dict)

    ############################
    @classmethod
    def _from_dict(cls, data):
        return cls(**data)

    @classmethod
    def _from_array(cls, data):
        return cls(
            x=data[0],
            u=data[1],
            t=data[2],
            dx=data[3],
            y=data[4]
    )

    @classmethod
    def load(cls, name):
        try:
            # try to load as new format (np.savez)
            with np.load(name) as data:
                return cls._from_dict(data)

        except ValueError:
            # If that fails, try to load as "legacy" numpy object array
            data = np.load(name, allow_pickle=True)
            return cls._from_array(data)


    ############################
    def _compute_size(self):

        self.time_final = self.t.max()
        self.time_steps = self.t.size

        self.n = self.time_steps
        self.m = self.u_sol.shape[1]

        # Check consistency between signals
        for arr in [self.x_sol, self.y_sol, self.u_sol, self.dx_sol]:
            if (arr is not None) and (arr.shape[0] != self.n):
                raise ValueError("Result arrays must have same length along axis 0")

    ############################
    def t2u(self, t ):
        """ get u from time """

        if t > self.time_final:
            raise ValueError("Got time t greater than final time")

            # Find time index
        i = (np.abs(self.t - t)).argmin()

        # Find associated control input
        u = self.u_sol[i,:]

        return u

    ############################
    def t2x(self, t ):
        """ get x from time """

        # Find time index
        i = (np.abs(self.t - t)).argmin()

        # Find associated control input
        x = self.x_sol[i,:]

        return x

    def plot(self, params):
        TrajectoryPlotter(self.sys).plot(self, params)

    def phase_plane_trajectory(self, x_axis, y_axis):
        TrajectoryPlotter(self.sys).phase_plane_trajectory(self, x_axis, y_axis)

class ClosedLoopTrajectory(Trajectory):
    """Trajectory with extra signals"""

    _dict_keys = ('x_sol', 'u_sol', 't', 'dx_sol', 'y_sol', 'r_sol')

    def __init__(self, x, u, t, dx, y, r):
        self.x_sol  = x
        self.u_sol  = u
        self.t  = t
        self.dx_sol = dx
        self.y_sol  = y
        self.r_sol = r

        self._compute_size()


class Simulator:
    """ 
    Simulation Class for open-loop ContinuousDynamicalSystem 
    --------------------------------------------------------
    ContinuousDynamicSystem : Instance of ContinuousDynamicSystem
    tf : final time
    n  : number of points
    solver : 'ode' or 'euler'
    """
    ############################
    def __init__(self, ContinuousDynamicSystem, tf=10, n=10001, solver='ode', x0=None):
        self.cds = ContinuousDynamicSystem
        self.t0 = 0
        self.tf = tf
        self.n  = int(n)
        self.dt = ( tf + 0.0 - self.t0 ) / ( n - 1 )
        self.solver = solver
        self.x0 = x0

        if self.x0 is None:
            self.x0 = np.zeros( self.cds.n )


    ##############################
    def compute(self):
        """ Integrate trought time """

        t  = np.linspace( self.t0 , self.tf , self.n )

        if self.solver == 'ode':
            x_sol = odeint( self.cds.fbar , self.x0 , t)

            # Compute inputs-output values
            y_sol = np.zeros(( self.n , self.cds.p ))
            u_sol = np.zeros((self.n,self.cds.m))
            dx_sol = np.zeros((self.n,self.cds.n))

            for i in range(self.n):
                xi = x_sol[i,:]
                ui = self.cds.ubar
                ti = t[i]

                dx_sol[i,:] = self.cds.f( xi , ui , ti )
                y_sol[i,:]  = self.cds.h( xi , ui , ti )
                u_sol[i,:]  = ui

        elif self.solver == 'euler':

            x_sol = np.zeros((self.n,self.cds.n))
            dx_sol = np.zeros((self.n,self.cds.n))
            u_sol = np.zeros((self.n,self.cds.m))
            y_sol = np.zeros((self.n,self.cds.p))

            # Initial State
            x_sol[0,:] = self.x0
            dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
            for i in range(self.n):

                xi = x_sol[i,:]
                ui = self.cds.ubar
                ti = t[i]

                if i+1<self.n:
                    dx_sol[i] = self.cds.f( xi , ui , ti )
                    x_sol[i+1,:] = dx_sol[i]*dt + xi

                y_sol[i,:] = self.cds.h( xi , ui , ti )
                u_sol[i,:] = ui

        sol = Trajectory(
            x=x_sol,
            u=u_sol,
            t=t,
            dx=dx_sol,
            y=y_sol
        )

        return sol


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

        cltraj = ClosedLoopTrajectory(
            x= sol.x_sol,
            u= u,
            t= sol.t,
            dx=sol.dx_sol,
            y=sol.y_sol,
            r=r,
        )

        return cltraj

    ###########################################################################
    def _compute_inputs(self, sol):
        """ Compute internal control signal of the closed-loop system """

        r_sol = sol.u_sol.copy() # reference is input of combined sys
        u_sol = np.zeros((self.n,self.sys.m))

        # Compute internal input signal
        for i in range(self.n):

            ri = r_sol[i,:]
            yi = sol.y_sol[i,:]
            ti = sol.t[i]

            u_sol[i,:] = self.ctl.c( yi , ri , ti )

        return (r_sol, u_sol)

'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass