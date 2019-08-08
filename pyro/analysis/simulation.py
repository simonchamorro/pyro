# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import odeint

##################################################################### #####
# Simulation Objects
##########################################################################

class Trajectory() :
    """Simulation data"""
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
        self.t_sol  = t
        self.dx_sol = dx
        self.y_sol  = y

        self._compute_size()

    ############################
    def save(self, name = 'trajectory_solution.npy' ):

        data = np.array( [ self.x_sol ,
                           self.u_sol ,
                           self.t_sol ,
                           self.dx_sol,
                           self.y_sol ] )

        np.save( name , data )

    ############################
    @staticmethod
    def load(name):

        data = np.load( name )
        tj = Trajectory(
            x  = data[0],
            u  = data[1],
            t  = data[2],
            dx = data[3],
            y  = data[4],
        )

        return tj

    ############################
    def _compute_size(self):

        self.time_final = self.t_sol.max()
        self.time_steps = self.t_sol.size

        self.n = self.time_steps
        self.m = self.u_sol.shape[1]

        # Check consistency between signals
        for arr in [self.x_sol, self.y_sol, self.u_sol, self.dx_sol]:
            if arr.size[0] != self.n:
                raise ValueError("Result arrays must have same length along axis 0")

    ############################
    def t2u(self, t ):
        """ get u from time """

        if t < self.time_final:
            # Find time index
            i = (np.abs(self.t_sol - t)).argmin()

            # Find associated control input
            u = self.u_sol[i,:]

        return u

    ############################
    def t2x(self, t ):
        """ get x from time """

        # Find time index
        i = (np.abs(self.t_sol - t)).argmin()

        # Find associated control input
        x = self.x_sol[i,:]

        return x


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
    def __init__(self, ContinuousDynamicSystem, tf=10, n=10001, solver='ode'):
        self.cds = ContinuousDynamicSystem
        self.t0 = 0
        self.tf = tf
        self.n  = int(n)
        self.dt = ( tf + 0.0 - self.t0 ) / ( n - 1 )
        self.x0 = np.zeros( self.cds.n )
        self.solver = solver


    ##############################
    def compute(self):
        """ Integrate trought time """
        
        self.t  = np.linspace( self.t0 , self.tf , self.n )
        self.dt = ( self.tf + 0.0 - self.t0 ) / ( self.n - 1 )
        
        self.J  = 0
        
        if self.solver == 'ode':
        
            self.x_sol = odeint( self.cds.fbar , self.x0 , self.t)   

            # Compute inputs-output values
            self.y_sol = np.zeros(( self.n , self.cds.p ))  
            self.u_sol = np.zeros((self.n,self.cds.m))
            
            for i in range(self.n):

                x = self.x_sol[i,:]  
                u = self.cds.ubar
                t = self.t[i]
                
                self.y_sol[i,:] = self.cds.h( x , u , t )
                self.u_sol[i,:] = u
                
        elif self.solver == 'euler':
            
            self.x_sol = np.zeros((self.n,self.cds.n))
            self.u_sol = np.zeros((self.n,self.cds.m))
            self.y_sol = np.zeros((self.n,self.cds.p))
            
            # Initial State    
            self.x_sol[0,:] = self.x0
            
            for i in range(self.n):
                
                x = self.x_sol[i,:]
                u = self.cds.ubar
                t = self.t[i]
                
                if i+1<self.n:
                    self.x_sol[i+1,:] = self.cds.f( x , u , t ) * self.dt + x
                
                self.y_sol[i,:] = self.cds.h( x , u , t )
                self.u_sol[i,:] = u


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
    def __init__(self, CLSystem , tf = 10 , n = 10001 , solver = 'ode' ):
        super().__init__(self, CLSystem , tf, n, solver)

        self.sys = CLSystem.sys
        self.ctl = CLSystem.ctl

    ###########################################################################
    def compute(self):
        
        super().compute()
        
        self.compute_inputs()
        
    ###########################################################################
    def compute_inputs(self):
        """ Compute internal control signal of the closed-loop system """
        
        self.r_sol = self.u_sol.copy() # reference is input of combined sys
        self.u_sol = np.zeros((self.n,self.sys.m))
        
        # Compute internal input signal
        for i in range(self.n):
            
            r = self.r_sol[i,:] 
            y = self.y_sol[i,:] 
            t = self.t[i]
            
            self.u_sol[i,:] = self.ctl.c( y , r , t )


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass