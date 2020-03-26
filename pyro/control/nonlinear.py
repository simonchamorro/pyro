# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:46:14 2018

@author: alxgr
"""

###############################################################################
import numpy as np
from scipy.interpolate import interp1d
###############################################################################
from pyro.control import controller
from pyro.dynamic import mechanical
###############################################################################




###############################################################################
# Computed Torque
###############################################################################
        
class ComputedTorqueController( controller.StaticController ) :
    """ 
    Inverse dynamic controller for mechanical system

    """    
    
    ############################
    def __init__(self, model = mechanical.MechanicalSystem() , traj = None ):
        """
        
        ---------------------------------------
        r  : reference signal vector  k x 1
        y  : sensor signal vector     p x 1
        u  : control inputs vector    m x 1
        t  : time                     1 x 1
        ---------------------------------------
        u = c( y , r , t )
        
        """
        
        self.model = model
        
        # Dimensions
        self.k = model.dof   
        self.m = model.m
        self.p = model.p
        
        controller.StaticController.__init__(self, self.k, self.m, self.p)
        
        # Label
        self.name = 'Computed Torque Controller'
        
        # Params
        self.w0   = 1
        self.zeta = 0.7 
        
        # Mode
        if traj == None:
            self.c = self.c_fixed_goal
        else:
            self.load_trajectory( traj )
            self.mode = 'interpol'
            self.c = self.c_trajectory_following
        
    
    #############################
    def c_fixed_goal( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
                
        x   = y 
        q_d = r
        
        u = self.fixed_goal_ctl( x , q_d , t )
        
        return u
    
        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        [ q , dq ]     = self.model.x2q( x )  
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )

        ddq_r          = self.compute_ddq_r( ddq_d , dq_d , q_d , dq , q )
        
        u              = self.model.actuator_forces( q , dq , ddq_r )
        
        return u
        
        
    ############################
    def compute_ddq_r( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state, compute ddq_r
        
        """
        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        ddq_r = ddq_d - 2 * self.zeta * self.w0 * dq_e - self.w0 ** 2 * q_e
        
        return ddq_r
    
        
    ############################
    def load_trajectory( self , traj  ):
        """ 
        
        Load Open-Loop trajectory solution to use as reference trajectory
        
        """
        
        self.trajectory = traj
        
        q   = traj.x_sol[ :,    0           :     self.model.dof ]
        dq  = traj.x_sol[ :, self.model.dof : 2 * self.model.dof ]
        ddq = traj.dx_sol[:, self.model.dof : 2 * self.model.dof ]
        t   = traj.t_sol
        
        # Create interpol functions
        self.q   = interp1d(t,q.T)
        self.dq  = interp1d(t,dq.T)
        self.ddq = interp1d(t,ddq.T)
        
        
    ############################
    def get_traj( self , t  ):
        """ 
        
        Find closest point on the trajectory
        
        """
        
        if t < self.trajectory.time_final :

            # Load trajectory
            q     = self.q(   t )
            dq    = self.dq(  t )
            ddq   = self.ddq( t )          

        else:
            
            q     = self.rbar
            dq    = np.zeros( self.model.dof )
            ddq   = np.zeros( self.model.dof )
        
        return ddq , dq , q
    
    
    ############################
    def traj_following_ctl( self , x , t = 0 ):
        """ 
        
        Given desired loaded trajectory and actual state, compute torques
        
        """
        
        [ q , dq ]         = self.model.x2q( x ) 
        
        ddq_d , dq_d , q_d = self.get_traj( t )

        ddq_r              = self.compute_ddq_r( ddq_d , dq_d , q_d , dq , q )
        
        u                  = self.model.actuator_forces( q , dq , ddq_r )
        
        return u
        
        
    #############################
    def c_trajectory_following( self , y , r , t ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        x = y 
        
        u = self.traj_following_ctl( x , t )
        
        
        return u
    


##############################################################################
        
class SlidingModeController( ComputedTorqueController ):
    """ 
    Sliding Mode Controller for fully actuated mechanical systems
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        ComputedTorqueController.__init__( self , model , traj )
        
        
        # Params
        
        self.lam  = 1   # Sliding surface slope
        self.gain = 1   # Discontinuous gain
        self.nab  = 0.1 # Min convergence rate
        
        
        
        
    ############################
    def compute_sliding_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def K( self , q , t ):
        """ Discontinuous gain matrix """
        
        dist_max = np.diag( np.ones( self.model.dof ) ) * self.gain
        conv_min = np.diag( np.ones( self.model.dof ) ) * self.nab
        
        K = dist_max + np.dot( self.model.H( q ) , conv_min ) 
        
        return K
        
        
    ############################
    def sliding_torque( self , ddq_r , s , dq , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = self.model.actuator_forces( q , dq , ddq_r )
        
        u_discontinuous = np.dot( self.K( q , t ) ,  np.sign( s ) )
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
        
        
    ############################
    def traj_following_ctl( self , x , t = 0 ):
        """ 
        
        Given desired loaded trajectory and actual state, compute torques
        
        """
        
        [ q , dq ]            = self.model.x2q( x ) 
        
        ddq_d , dq_d , q_d    = self.get_traj( t )

        [ s , dq_r , ddq_r ]  = self.compute_sliding_variables( ddq_d , dq_d , 
                                                                q_d , dq , q )
        
        u                     = self.sliding_torque( ddq_r , s , dq , q , t )
        
        return u
        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )

        [ s , dq_r , ddq_r ]  = self.compute_sliding_variables( ddq_d , dq_d , 
                                                                q_d , dq , q )
        
        u                     = self.sliding_torque( ddq_r , s , dq , q , t )
        
        return u

##############################################################################
        
class AdaptativeController( ComputedTorqueController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        ComputedTorqueController.__init__( self , model , traj )
        
        self.name = 'Adaptive controller'
        
        # Params
        self.dt = 0.001
        self.A = np.array([0.2,0.2])
        self.T=np.eye(2)
        #self.A[0] = 2
        #self.A[1] = 5
        self.Kd = 1
        self.lam  = 1   # Sliding surface slope
        self.nab  = 0.1 # Min convergence rate
        
        
        
        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def Adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = self.Kd*s
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
                        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        

        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        
        Y = np.zeros(2)
        dA = np.zeros(2)
        #T = np.zeros((2,2))
        #T[0,0] = 2
        #T[1,1] = 4
        
        
        Y[0]=ddq_r
        Y[1]=np.sin(q)
        
        #print(Y)
        #print(s)
        b = Y * s
        dA=-1*np.dot( self.T , b )
        
        self.A=self.A+dA*self.dt
        #print(self.A)
                
        u                     = self.Adaptative_torque( Y , s  , q , t )
        
        return u
    
##############################################################################
        
class AdaptativeController_2( ComputedTorqueController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        ComputedTorqueController.__init__( self , model , traj )
        
        self.name = 'Adaptive controller'
        
        # Params
        self.dt = 0.001
        self.A = np.zeros(5)
        self.T=np.eye(5)
        self.Kd = np.eye(2)
        self.lam  = 1   # Sliding surface slope
        
        
        
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin usefull in other computation 
        ------------------------------------------------
        
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        
        return [c1,s1,c2,s2,c12,s12]

        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def Adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = np.dot(self.Kd,s)
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
                        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        

        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        Y = np.zeros((2,5))
        dA = np.zeros(5)
                
        Y[0,0]=ddq_r[0]*c2
        Y[0,1]=ddq_r[1]*c2
        Y[0,2]=s2*dq[1]*dq_r[0]
        Y[0,3]=s2*(dq[0]+dq[1])*dq_r[1]
        Y[0,4]=s1+s12
        Y[1,0]=ddq_r[0]*c2
        Y[1,1]=ddq_r[1]
        Y[1,2]=s2*dq[0]*dq_r[0]
        Y[1,3]=0
        Y[1,4]=s12
        
        b = np.dot(Y.T,s)
        dA=-1*np.dot( self.T , b )
        
        self.A=self.A+dA*self.dt
        #print(self.A)
                
        u                     = self.Adaptative_torque( Y , s  , q , t )
        
        return u

##############################################################################
        
class AdaptativeController_WCRT( ComputedTorqueController ):
    """ 
    Adaptative Controller for fully actuated mechanical systems (single pendulum)
    """
    
    
    ############################
    def __init__( self , model , traj = None ):
        """ """
        
        ComputedTorqueController.__init__( self , model , traj )
        
        self.name = 'Adaptive controller'
        
        # Params
        self.dt = 0.001
        self.A = np.zeros(6)
        self.T=np.eye(6)
        self.Kd = np.eye(3)
        self.lam  = 1   # Sliding surface slope
        
        
        
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin usefull in other computation 
        ------------------------------------------------
        
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c3  = np.cos( q[2] )
        s3  = np.sin( q[2] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c23,s23]

        
    ############################
    def adaptative_variables( self , ddq_d , dq_d , q_d , dq , q ):
        """ 
        
        Given desired trajectory and actual state
        
        """        
        q_e   = q  -  q_d
        dq_e  = dq - dq_d
        
        s      = dq_e  + self.lam * q_e
        dq_r   = dq_d  - self.lam * q_e
        ddq_r  = ddq_d - self.lam * dq_e
        
        return [ s , dq_r , ddq_r ]
        
        
    ############################
    def Adaptative_torque( self , Y , s , q , t ):
        """ 
        
        Given actual state, compute torque necessarly to guarantee convergence
        
        """
                
        u_computed      = np.dot( Y , self.A  )
        
        u_discontinuous = np.dot(self.Kd,s)
        
        u_tot = u_computed - u_discontinuous
        
        return u_tot
                        
        
    ############################
    def fixed_goal_ctl( self , x , q_d , t = 0 ):
        """ 
        
        Given desired fixed goal state and actual state, compute torques
        
        """
        
        [ q , dq ]     = self.model.x2q( x ) 
        
        ddq_d          =   np.zeros( self.model.dof )
        dq_d           =   np.zeros( self.model.dof )
        

        [ s , dq_r , ddq_r ]  = self.adaptative_variables( ddq_d , dq_d , 
                                                           q_d , dq , q )
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        Y = np.zeros((3,6))
        dA = np.zeros(5)
                
        Y[0,0]=ddq_r[0]
        Y[0,1]=ddq_r[1]*(s2+s2*c3)
        Y[0,2]=ddq_r[2]*s23
        Y[0,3]=(dq[2]*s2*s3+dq[1]*c2+dq[1]*c2*c3)*dq_r[1]
        Y[0,4]=(dq[2]+dq[1])*c23*dq_r[2]
        Y[0,5]=0
        Y[1,0]=ddq_r[0]*(s2+s2*c3)
        Y[1,1]=ddq_r[1]*(1+c3+c3**2)
        Y[1,2]=ddq_r[2]*(c3+c3**2)
        Y[1,3]=(dq[2]*(s3+s3*c3))*dq_r[1]
        Y[1,4]=(dq[2]*(s3+s3*c3)+dq[1]*(s3+s3*c3)+dq[0]*(c2*c3))*dq_r[2]
        Y[1,5]=c2+c23
        Y[2,0]=ddq_r[0]*s23
        Y[2,1]=ddq_r[1]*(c3+c3**2)
        Y[2,2]=ddq_r[2]
        Y[2,3]=(dq[1]*(s3+s3*c3)+dq[1]*c2*c3)*dq_r[1]
        Y[2,4]=0
        Y[2,5]=c23
        
        b = np.dot(Y.T,s)
        dA=-1*np.dot( self.T , b )
        
        self.A=self.A+dA*self.dt
        print(self.A)
                
        u                     = self.Adaptative_torque( Y , s  , q , t )
        
        return u
        
##############################################################################
    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic import pendulum
    
    sp = pendulum.SinglePendulum()
    c  = ComputedTorqueController( sp )
    
    # New cl-dynamic
    clsp = controller.ClosedLoopSystem( sp ,  c )
    
    x0 = np.array([2,0])
    clsp.plot_phase_plane_trajectory( x0 )
    clsp.sim.plot('xu')
    clsp.animate_simulation()
    
    ####################################
    dp = pendulum.DoublePendulum()
    c2  = ComputedTorqueController( dp )
    
    # New cl-dynamic
    cldp = controller.ClosedLoopSystem( dp ,  c2 )
    
    x0 = np.array([2,1,0,0])
    cldp.plot_phase_plane_trajectory( x0 , 10 , 0 , 2)
    cldp.plot_phase_plane_trajectory( x0 , 10 , 1 , 3)
    cldp.sim.plot('xu')
    cldp.animate_simulation()
        
