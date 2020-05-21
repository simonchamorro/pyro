#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:51:49 2020

@author: alex
"""

##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic.manipulator import SpeedControlledManipulator
from pyro.dynamic.manipulator import ThreeLinkManipulator3D
##############################################################################
#For robot3DOF
from pyro.dynamic import manipulator

########################################
# Model de robot cinématique seulement
########################################

class Robot1( SpeedControlledManipulator ):
    """ 
    Robot
    """
    ###########################
    def __init__(self):
        """ """
        
        # Dimensions
        dof = 3
        e   = 2
               
        # initialize standard params
        SpeedControlledManipulator.__init__( self, dof, e)
        
        # Name
        self.name = 'Robot pour partie II de la problématique'
        
        l1  = 1.2
        l2  = 0.5
        l3  = 0.5
        
        self.l = np.array([l1,l2,l3])
        
        
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
        
        cos_rel = np.array( [ c1 , c2 , c3 ])
        sin_rel = np.array( [ s1 , s2 , s3 ])
        
        c12    = np.cos( q[0] + q[1] )
        s12    = np.sin( q[0] + q[1] )
        c123   = np.cos( q[0] + q[1] + q[2])
        s123   = np.sin( q[0] + q[1] + q[2])
        
        cos_abs = np.array( [ c1 , c12 , c123 ])
        sin_abs = np.array( [ s1 , s12 , s123 ])
        
        return [cos_rel,sin_rel,cos_abs,sin_abs]
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ """
        
        [cos_rel,sin_rel,cos_abs,sin_abs] = self.trig( q )
        
        x = (self.l * sin_abs).sum()
        y = (self.l * cos_abs).sum()
        
        r = np.array([x,y])
        
        return r
    
    
    ##############################
    def J(self, q ):
        """ """
        
        [cos_rel,sin_rel,cos_abs,sin_abs] = self.trig( q )
        
        J = np.zeros( ( self.e  , self.dof ) ) # Place holder
        
        J[0,0] = self.l[2] * cos_abs[2] + self.l[1] * cos_abs[1]  + self.l[0] * cos_abs[0] 
        J[0,1] = self.l[2] * cos_abs[2] + self.l[1] * cos_abs[1]             
        J[0,2] = self.l[2] * cos_abs[2]           

        
        J[1,0] = - self.l[2] * sin_abs[2] - self.l[1] * sin_abs[1]  - self.l[0] * sin_abs[0] 
        J[1,1] = - self.l[2] * sin_abs[2] - self.l[1] * sin_abs[1]             
        J[1,2] = - self.l[2] * sin_abs[2]           
        
        return J
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 3
        
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
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        
        ###########################
        # robot kinematic
        ###########################
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [cos_rel,sin_rel,cos_abs,sin_abs] = self.trig( q )
        
        pts[1,0] = self.l[0] * sin_abs[0]
        pts[2,0] = self.l[1] * sin_abs[1] + pts[1,0]
        pts[3,0] = self.l[2] * sin_abs[2] + pts[2,0]

        
        pts[1,1] = self.l[0] * cos_abs[0]
        pts[2,1] = self.l[1] * cos_abs[1] + pts[1,1]
        pts[3,1] = self.l[2] * cos_abs[2] + pts[2,1]

        
        lines_pts.append( pts )
                
        return lines_pts
    
    
    
    
class Robot2( ThreeLinkManipulator3D ):
    """ 
    Robot 2 for drilling
    """
    
    
    ##############################
    def f_ext(self, q , dq , t = 0 ):
        """ 
        External force due to contact during drilling
        
        """
        
        r = self.forward_kinematic_effector( q )
        
        dr = self.forward_differential_kinematic_effector(q, dq)
        
        # Contact:
        if r[2] < 1 :
            
            fx = - dr[0] * 1000 # damping lateral
            fy = - dr[1] * 1000 # damping lateral
            fz = - dr[2] * 100 # damping vertical 
            
            f_ext = np.array([fx,fy,fz])
            
        else:
            
            f_ext = np.zeros( self.e )
        
        return f_ext
    
    
    
    

class Robot3( SpeedControlledManipulator ):
    """ 
    Robot
    """
    
    ############################
    def __init__(self):
        
        SpeedControlledManipulator.__init__( self, 3, 3)
    
        # Kinematic
        self.l1  = 1 
        self.l2  = 1
        self.l3  = 1
        self.lc1 = 1
        self.lc2 = 1
        self.lc3 = 1
        
        # Total length
        self.lw  = (self.l1+self.l2+self.l3)
    
    ##############################
    def trig(self, q ):
        """ 
        Compute cos and sin 
        --------------------
        """
        
        c1  = np.cos( q[0] )
        s1  = np.sin( q[0] )
        c2  = np.cos( q[1] )
        s2  = np.sin( q[1] )
        c3  = np.cos( q[2] )
        s3  = np.sin( q[2] )
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23]
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # End-effector
        z3 = self.l1 - self.l2 * s2 - self.l3 * s23
        
        r3 = self.l2 * c2 + self.l3 * c23
        x3 = r3 * c1
        y3 = r3 * s1
                
        r = np.array([x3, y3, z3])
        
        return r
    
    
    ##############################
    def J(self, q ):
        """ """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        J = np.zeros((3,3))
        
        J[0,0] =  -( self.l2 * c2 + self.l3 * c23 ) * s1
        J[0,1] =  -( self.l2 * s2 + self.l3 * s23 ) * c1
        J[0,2] =  - self.l3 * s23 * c1
        J[1,0] =   ( self.l2 * c2 + self.l3 * c23 ) * c1
        J[1,1] =  -( self.l2 * s2 + self.l3 * s23 ) * s1
        J[1,2] =  - self.l3 * s23 * s1
        J[2,0] =  0
        J[2,1] =  -( self.l2 * c2 + self.l3 * c23 )
        J[2,2] =  - self.l3 * c23
        
        return J
    
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 2
        
        domain  = [ (-l,l) , (-l,l) , (0,l*2) ]#  
                
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
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 5 , 3 ))
        pts[0,:] = np.array([-1,-1,0])
        pts[1,:] = np.array([+1,-1,0])
        pts[2,:] = np.array([+1,+1,0])
        pts[3,:] = np.array([-1,+1,0])
        pts[4,:] = np.array([-1,-1,0])
        
        lines_pts.append( pts )
        
        ###########################
        # robot kinematic
        ###########################
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # Three robot points

        # Shperical point 
        pts[1,0] = 0
        pts[1,1] = 0
        pts[1,2] = self.l1 
        
        # Elbow
        z2 = self.l1 - self.l2 * s2
        
        r2 = self.l2 * c2
        x2 = r2 * c1
        y2 = r2 * s1
        
        pts[2,0] = x2
        pts[2,1] = y2
        pts[2,2] = z2
        
        # End-effector
        z3 = self.l1 - self.l2 * s2 - self.l3 * s23
        
        r3 = self.l2 * c2 + self.l3 * c23
        x3 = r3 * c1
        y3 = r3 * s1
                
        pts[3,0] = x3
        pts[3,1] = y3
        pts[3,2] = z3 
        
        lines_pts.append( pts )
                
        return lines_pts
    
class robot3DOF( manipulator.ThreeLinkManipulator3D ):
    """ 
    Three link Manipulator Class 
    -------------------------------
    
    base:     revolute arround z
    shoulder: revolute arround y
    elbow:    revolute arround y
    
    see Example 4.3 in
    http://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-manipdyn_v1_2.pdf
    """
    
    ############################
    def __init__(self):
        """ """
        
       
               
        # initialize standard params
        manipulator.ThreeLinkManipulator3D.__init__(self)
        
        # Name
        self.name = 'robot3DOF'
        
        # params
        self.setparams()
                
            
    #############################
    def setparams(self):
        """ Set model parameters here """
        
        # kinematic
        self.l1  = 0.3
        self.l2  = 0.525 
        self.l3  = 0.375
        
        
        # dynamic
        self.I1        = .66125
        self.m2       = 1.589
        self.m3       = 0.545
        
        self.gravity = 9.81
        
        
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
        c12 = np.cos( q[0] + q[1] )
        s12 = np.sin( q[0] + q[1] )
        c23 = np.cos( q[2] + q[1] )
        s23 = np.sin( q[2] + q[1] )
        
        return [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23]
    
    
    ##############################
    def forward_kinematic_effector(self, q ):
        """ """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        lz1     = self.l1
        lx2     = self.l2
        lx3     = self.l3
        
        
        # Three robot points
        
        # Base of the robot
        p0 = [0,0,0]
        
        # Shperical point 
        p1 = [ 0, 0, lz1 ]
        
        # Elbow
        
        
        x2 = lx2*c1*c2
        y2 = lx2*c1*s1
        z2 = lz1 + lx2*s2
        
        p2 = [ x2, y2, z2 ]
        
        # End-effector

        x3 = c1*(lx3*c23 + lx2*c2)
        y3 = s1*(lx3*c23 + lx2*c2)
        z3 = lz1 + lx3*s23 + lx2*s2
                
        r = np.array([x3, y3, z3])
        
        return r
    
    
    ##############################
    def J(self, q ):
        """ """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        
        sin=np.sin
        cos=np.cos
        
        lz1     = self.l1
        lx2     = self.l2
        lx3     = self.l3
        
        J = np.zeros((3,3))
        
        J[0,0] =  -sin(q1)*(lx3*cos(q2 + q3) + lx2*cos(q2))
        J[0,1] =  -cos(q1)*(lx3*sin(q2 + q3) + lx2*sin(q2))
        J[0,2] =  -lx3*sin(q2 + q3)*cos(q1)
        
        J[1,0] =   cos(q1)*(lx3*cos(q2 + q3) + lx2*cos(q2))
        J[1,1] =  -sin(q1)*(lx3*sin(q2 + q3) + lx2*sin(q2))        
        J[1,2] =  -lx3*sin(q2 + q3)*sin(q1)
        
        J[2,0] =  0
        J[2,1] =  lx3*cos(q2 + q3) + lx2*cos(q2)
        J[2,2] =  lx3*cos(q2 + q3)
        
        return J
    
    
    ##############################
    def f_ext(self, q , dq , t = 0 ):
        """ """
        
        f_ext = np.zeros( self.e ) # Default is zero vector
        
        return f_ext
        
    
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq**T * H(q) * dq
        
        """  
        
        [c1,s1,c2,s2,c3,s3,c23,s23] = self.trig( q )
        
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        
        sin=np.sin
        cos=np.cos
        
        lz1     = self.l1
        lx2     = self.l2
        lx3     = self.l3
        
        I1      = self.I1
        m2      = self.m2
        m3      = self.m3
        
        
        
        H = np.zeros((3,3))
        
        H[0,0] = I1 + (m3*(2*(lx2*cos(q2)*sin(q1) + lx3*cos(q2)*cos(q3)*sin(q1) - lx3*sin(q1)*sin(q2)*sin(q3))**2 + 2*(lx2*cos(q1)*cos(q2) + lx3*cos(q1)*cos(q2)*cos(q3) - lx3*cos(q1)*sin(q2)*sin(q3))**2))/2 + lx2**2*m2*cos(q2)**2
        H[1,0] = 0        
        H[2,0] = 0
        
        H[0,1] = H[1,0]
        H[1,1] = (m3*(2*lx2**2 + 4*cos(q3)*lx2*lx3 + 2*lx3**2))/2 + lx2**2*m2
        H[2,1] = m3*(lx3**2 + lx2*cos(q3)*lx3)
        
        H[0,2] = H[2,0]
        H[1,2] = H[2,1]
        H[2,2] = lx3**2*m3        

        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
         Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C**T
        
        
        """ 
        
        [c1,s1,c2,s2,c12,s12] = self.trig( q )
        
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        
        sin=np.sin
        cos=np.cos
        
        q1d     = dq[0]
        q2d     = dq[1]
        q3d     = dq[2]
        
        lz1     = self.l1
        lx2     = self.l2
        lx3     = self.l3
        
        I1      = self.I1
        m2      = self.m2
        m3      = self.m3
        
        
        
        C = np.zeros((3,3))
        
        C[0,0] = 0
        C[0,1] = -q1d*(lx3**2*m3*sin(2*q2 + 2*q3) + lx2**2*m2*sin(2*q2) + lx2**2*m3*sin(2*q2) + 2*lx2*lx3*m3*sin(2*q2 + q3))
        C[0,2] = -lx3*m3*q1d*(lx3*sin(2*q2 + 2*q3) + lx2*sin(q3) + lx2*sin(2*q2 + q3))
        
        C[1,0] = (q1d*(lx3**2*m3*sin(2*q2 + 2*q3) + lx2**2*m2*sin(2*q2) + lx2**2*m3*sin(2*q2) + 2*lx2*lx3*m3*sin(2*q2 + q3)))/2
        C[1,1] = 0
        C[1,2] = -lx2*lx3*m3*sin(q3)*(2*q2d + q3d)
        
        C[2,0] = lx3*m3*q1d*((lx3*sin(2*q2 + 2*q3))/2 + (lx2*sin(q3))/2 + (lx2*sin(2*q2 + q3))/2)
        C[2,1] = (lx2*lx3*m3*sin(q3)*(2*q2d + q3d))/2
        C[2,2] = -(lx2*lx3*m3*q2d*sin(q3))/2 
        
        return C
    
    
    ###########################################################################
    def B(self, q ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.diag( np.ones( self.dof ) ) #  identity matrix
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        
        sin=np.sin
        cos=np.cos
        
        lz1     = self.l1
        lx2     = self.l2
        lx3     = self.l3
        
        I1      = self.I1
        m2      = self.m2
        m3      = self.m3
        
        G = np.zeros(3)
        
        
        g = self.gravity
        

        
        G[0] = 0
        G[1] = g*(m3*(lx3*cos(q2 + q3) + lx2*cos(q2)) + lx2*m2*cos(q2))
        G[2] = g*lx3*m3*cos(q2 + q3)

        return G
    
        
    ###########################################################################
    def d(self, q , dq ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        D = np.zeros((3,3))
        
        d = np.dot( D , dq )
        
        return d
    
        
    ###########################################################################
    # Graphical output
    ###########################################################################
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = 2
        
        domain  = [ (-l,l) , (-l,l) , (0,l*2) ]#  
                
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
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 5 , 3 ))
        pts[0,:] = np.array([-1,-1,0])
        pts[1,:] = np.array([+1,-1,0])
        pts[2,:] = np.array([+1,+1,0])
        pts[3,:] = np.array([-1,+1,0])
        pts[4,:] = np.array([-1,-1,0])
        
        lines_pts.append( pts )
        
        ###########################
        # robot kinematic
        ###########################
        
        pts      = np.zeros(( 4 , 3 ))
        pts[0,:] = np.array([0,0,0])
        
        [c1,s1,c2,s2,c3,s3,c12,s12,c23,s23] = self.trig( q )
        
        # Three robot points

        lz1     = self.l1
        lx2     = self.l2
        lx3     = self.l3
        
        I1      = self.I1
        m2      = self.m2
        m3      = self.m3
        
        
        pts[1,0] = 0
        pts[1,1] = 0
        pts[1,2] = lz1
        
        pts[2,0] = lx2*c1*c2
        pts[2,1] = lx2*c1*s1
        pts[2,2] = lz1 + lx2*s2
        
        pts[3,0] = c1*(lx3*c23 + lx2*c2)
        pts[3,1] = s1*(lx3*c23 + lx2*c2)
        pts[3,2] = lz1 + lx3*s23 + lx2*s2
        
        lines_pts.append( pts )
                
        return lines_pts
    
    #Plot robot   
asimov=robot3DOF()
asimov.show3([0, 0, np.pi/2])