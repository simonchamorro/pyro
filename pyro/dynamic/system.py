# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 11:51:55 2015

@author: agirard
"""

import numpy as np

from pyro.analysis import simulation
from pyro.analysis import phaseanalysis
from pyro.analysis import graphical
       
'''
###############################################################################
'''


class ContinuousDynamicSystem:
    """ 
    Mother class for continuous dynamical systems
    ----------------------------------------------
    n : number of states
    m : number of control inputs
    p : number of outputs
    ---------------------------------------
    dx = f( x , u , t )
    y  = h( x , u , t )
    
    optionnal: 
    u = t2u( t ) : time-dependent input signal
    
    
    """
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    ############################
    def __init__(self, n = 1, m = 1, p = 1):
        """ 
        The __init__ method of the Mother class can be used to fill-in default
        labels, units, bounds, and base values.
        """
        
        #############################
        # Parameters
        #############################

        # Dimensions
        self.n = n   
        self.m = m   
        self.p = p
        
        # Labels
        self.name = 'ContinuousDynamicSystem'
        self.state_label  = []
        self.input_label  = []
        self.output_label = []
        
        # Units
        self.state_units  = []
        self.input_units  = []
        self.output_units = []
        
        # Default Label and units
        for i in range(n):
            self.state_label.append('State '+str(i))
            self.state_units.append('')
        for i in range(m):
            self.input_label.append('Input '+str(i))
            self.input_units.append('')
        for i in range(p):
            self.output_label.append('Output '+str(i))
            self.output_units.append('')
        
        # Default state and input domain
        self.x_ub = np.zeros(self.n) +10 # States Upper Bounds
        self.x_lb = np.zeros(self.n) -10 # States Lower Bounds
        self.u_ub = np.zeros(self.m) +1  # Control Upper Bounds
        self.u_lb = np.zeros(self.m) -1  # Control Lower Bounds
        
        # Default state and inputs values    
        self.xbar = np.zeros(self.n)
        self.ubar = np.zeros(self.m)
        
        ################################
        # Variables
        ################################
        
        # Initial value for simulations
        self.x0   = np.zeros(self.n) 
        
        # Last simulation memory
        # Only for easy acces in interactive mode
        # Do not use in method
        self.sim  = None   
        self.ani  = None
        self.traj = None
        self.pp   = None
        
    
    #############################
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
        
        ################################################
        # Place holder: put the equations of motion here
        ################################################
        
        raise NotImplementedError
        
        return dx
    
    
    ###########################################################################
    # The following functions can be overloaded when necessary by child classes
    ###########################################################################
    
    #############################
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
        
        y = x      # default output is all states
        
        return y
    
    #############################
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
        
        #Default is a constant signal
        u = self.ubar
        
        return u
    
        
    ###########################################################################
    # Basic domain checks, ovewload if something more complex is needed
    ###########################################################################
        
    #############################
    def isavalidstate(self , x ):
        """ check if x is in the state domain """
        ans = False
        for i in range(self.n):
            ans = ans or ( x[i] < self.x_lb[i] )
            ans = ans or ( x[i] > self.x_ub[i] )
            
        return not(ans)
        
    #############################
    def isavalidinput(self , x , u):
        """ check if u is in the control inputs domain given x """
        ans = False
        for i in range(self.m):
            ans = ans or ( u[i] < self.u_lb[i] )
            ans = ans or ( u[i] > self.u_ub[i] )
            
        return not(ans)
    
    
    ###########################################################################
    # Place holder graphical output, ovewload with specific graph output
    ###########################################################################
        
    #############################
    def xut2q( self, x , u , t ):
        """ Compute configuration variables ( q vector ) """
        
        # default is q = x
        
        return x
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 10
        
        domain  = [ (-l,l) , (-l,l) , (-l,l) ]#  
                
        return domain
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        
        ###########################
        # Your graphical code here
        ###########################
            
        # simple place holder
        for i in range(self.n):
            pts      = np.zeros(( 1 , 3 ))     # array of 1 pts for the line
            pts[0,0] = q[i]                    # x cord of point 0 = q
            lines_pts.append( pts )            # list of all arrays of pts
                
        return lines_pts
    
    ###########################################################################
    # No need to overwrite the following functions for custom dynamic systems
    ###########################################################################
    
    #############################
    def fsim( self, x , t ):
        """ 
        Continuous time foward dynamics evaluation dx = f(x,t), inlcuding the
        internal reference input signal computation
        
        INPUTS
        x  : state vector             n x 1
        t  : time                     1 x 1
        
        OUPUTS
        dx : state derivative vector  n x 1
        
        """
        
        u  = self.t2u( t )
        dx = self.f( x, u, t)
        
        return dx
    

    #############################
    def x_next( self , x , u , t , dt = 0.1 , steps = 1 ):
        """ 
        Discrete time foward dynamics evaluation 
        -------------------------------------
        - using Euler integration
        
        """
        
        x_next = np.zeros(self.n) # k+1 State vector
        
        # Multiple integration steps
        for i in range(steps):
        
            x_next = self.f(x,u,t) * dt + x
            
            # Multiple steps
            x =  x_next
        
        return x_next
    
    
    ###########################################################################
    # Quick Analysis Shorcuts
    ###########################################################################

    def get_plotter(self):
        return graphical.TrajectoryPlotter(self)

    def get_animator(self):
        return graphical.Animator(self)

    #############################
    def plot_phase_plane(self , x_axis = 0 , y_axis = 1 ):
        """ 
        Plot Phase Plane vector field of the system
        ------------------------------------------------
        x_axis : index of state on x axis
        y_axis : index of state on y axis
        
        """

        self.pp = phaseanalysis.PhasePlot( self , x_axis , y_axis )
        
        self.pp.plot()
        
        
    #############################
    def compute_trajectory(
        self, tf=10, n=10001, solver='ode', costfunc=None):
        """ 
        Simulation of time evolution of the system
        ------------------------------------------------
        x0 : initial time
        tf : final time
        
        """

        sim = simulation.Simulator(self, tf, n, solver)
        
        traj = sim.compute()

        if costfunc is not None:
            traj = costfunc.eval( traj )
        
        # Object memory
        self.traj = traj

        return traj


    #############################
    def plot_trajectory(self, plot='x', **kwargs):
        """
        Plot time evolution of a simulation of this system
        ------------------------------------------------

        """
        
        # Check is trajectory is already computed
        if self.traj == None:
            self.compute_trajectory()
        
        self.get_plotter().plot( self.traj, plot, **kwargs)


    #############################
    def plot_phase_plane_trajectory(self, x_axis=0, y_axis=1):
        """
        Plot a trajectory in the Phase Plane
        ---------------------------------------------------------------

        """
        
        # Check is trajectory is already computed
        if self.traj == None:
            self.compute_trajectory()
            
        self.get_plotter().phase_plane_trajectory( self.traj, x_axis , y_axis)


    #############################
    def plot_phase_plane_trajectory_3d(self ,  x_axis=0, y_axis=1, z_axis=2):
        """
        Simulates the system and plot the trajectory in the Phase Plane
        ---------------------------------------------------------------
        x0 : initial time
        tf : final time
        x_axis : index of state on x axis
        y_axis : index of state on y axis

        """
        
        # Check is trajectory is already computed
        if self.traj == None:
            self.compute_trajectory()
            
        self.get_plotter().phase_plane_trajectory_3d( self.traj, x_axis , y_axis, z_axis)


    #############################################
    def show(self, q , x_axis = 0 , y_axis = 1 ):
        """ Plot figure of configuration q """
        
        self.ani = graphical.Animator( self )
        self.ani.x_axis  = x_axis
        self.ani.y_axis  = y_axis
        
        self.ani.show( q )
        
    
    #############################################
    def show3(self, q ):
        """ Plot figure of configuration q """
        
        self.ani = graphical.Animator( self )
        
        self.ani.show3( q )

    ##############################
    def animate_simulation(self, **kwargs):
        """
        Show Animation of the simulation
        ----------------------------------
        time_factor_video < 1 --> Slow motion video

        """
        
        # Check is trajectory is already computed
        if self.traj == None:
            self.compute_trajectory()

        self.get_animator().animate_simulation( self.traj, **kwargs)


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass