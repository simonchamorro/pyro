# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:09:37 2017

@author: Charles.Khazoom@hotmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, rosen, rosen_der
from pyro.dynamic import manipulator
from pyro.planning import plan
from pyro.analysis import costfunction
from pyro.analysis import Trajectory
'''
################################################################################
'''


#class TorqueSquaredCostFunction( costfunction.QuadraticCostFunction ):
#    def __init__(self):
#        costfunction.QuadraticCostFunction.__init__( self )
#
#cost_fun = TorqueSquaredCostFunction()

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
res.x

res2 = minimize(rosen, x0, method='SLSQP', bounds=None, constraints=())

fun3 = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2


cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},{'type': 'eq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
# see this for further examples on contrained optimization problems : https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html 

x0_0 = [1.3, 0.7]
bnds = ((0, None), (0, None))
res3 = minimize(fun3, x0_0, method='SLSQP', bounds=bnds, constraints=cons,tol=1e-6)

'''
################################################################################
'''
# create system 
sys = manipulator.TwoLinkManipulator()

# PATCH : must compute trajectory before passing it to OpenLoopController() class constructor
# create open-loop controller
tf=4
sys.compute_trajectory( tf )

#ctl  = plan.OpenLoopController.load_from_file( 'double_pendulum_rrt.npy' )
ctl = plan.OpenLoopController(sys.traj)
ctl.trajectory.u=np.ones(sys.traj.u.shape)*10 # create inputs


sys.f(np.array([1,2,3,4]),np.array([3,2])) # compute f
sys.x0 =np.array([np.pi/2,np.pi/2,0,0])


# New cl-dynamic
cl_sys = ctl + sys
# cl_sys.m is 1 should be 2 ??

cl_sys.compute_trajectory( tf )
cl_sys.plot_trajectory('xu')

sys.cost_function.trajectory_evaluation(sys.traj)
#cl_sys.animate_simulation()

# 
'''
Create optization problem
'''

# set cost function
# minimize torque square is quadraic cost with Q and V set to 0
#class TorqueSquaredCostFunction( costfunction.QuadraticCostFunction ): should implement this class
sys.cost_function.Q=np.zeros(sys.cost_function.Q.shape)
sys.cost_function.V=np.zeros(sys.cost_function.V.shape)
sys.cost_function.R=np.ones(sys.cost_function.R.shape) # # cl_sys.m is 1 should be 2 ??


ngrid =50 # number of gridpoint

######## set bounds ##########
ub_t0 = 0 # bounds on t0 
lb_t0 = 0

ub_tf = 4 # bounds on tf 
lb_tf = 4

#ub_state = np.array([])
#lb_state
ub_x = sys.x_ub # bounds on x
lb_x = sys.x_lb

ub_u = sys.u_ub # bounds on inputs u
lb_u = sys.u_lb

ub_x0 = [0,0,0,0] # bounds on inital state
lb_x0 = [0,0,0,0]


ub_xF = [np.pi/2,np.pi/2,0,0]#sys.x_ub # bounds on final state
lb_xF = [np.pi/2,np.pi/2,0,0]

'''
create initial guess

'''

x_guess = np.linspace(ub_x0, ub_xF, ngrid)
u_guess = np.ones([ngrid,sys.m])
t_guess = np.linspace(ub_t0, ub_tf, ngrid)
y_guess= x_guess

#guess_traj = Trajectory(x_guess, u_guess, t_guess, dx_guess, y_guess)

######## set equality contraints (other than dynamics) ##########
# sys.f contains the dynamics ! 
'''
Convert to non-linear program (direct collocation)
'''

#dec_var = # decision variables
#optim_traj = Trajectory(x_opt, u_opt, t_opt, dx_opt, y_opt)
#lb_dec_var = ub_x*ones(x.shape)

'''
Solve non-linear program
'''

'''
Interpolate solution in trajectory object
'''