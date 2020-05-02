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