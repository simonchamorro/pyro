# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:09:37 2017

@author: Charles.Khazoom@hotmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, rosen, rosen_der#, interpolate
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

def f(x):
    #return (x[0] - 1)**2 + (x[1] - 2.5)**2
    return x[0] - 2 * x[1] + 2

cons = ({'type': 'eq', 'fun': f},{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
# see this for further examples on contrained optimization problems : https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html 

x0_0 = [1.3, 0.7]

bnds_arr = np.array([[0,None],[0,None]])
bnds=tuple(map(tuple, bnds_arr))
#bnds = ((0, None), (0, None))


res3 = minimize(fun3, x0_0, method='SLSQP', bounds=bnds, constraints=cons,tol=1e-6)
res3
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
sys.cost_function.R=np.ones(sys.cost_function.R.shape) #

global ngrid
ngrid =50 # number of gridpoint

######## set bounds ##########
ub_t0 = 0 # bounds on t0 
lb_t0 = 0

ub_tf = 4 # bounds on tf 
lb_tf = 4
# should implement an error message if tf is not >t0 and in t0<0

#ub_state = np.array([])
#lb_state
sys.x_ub=[2*np.pi,2*np.pi,None,None]
sys.x_lb=[-2*np.pi,-2*np.pi,None,None]

ub_x = sys.x_ub # bounds on x
lb_x = sys.x_lb

ub_u = sys.u_ub # bounds on inputs u
lb_u = sys.u_lb

ub_x0 = [0,0,0,0] # bounds on inital state
lb_x0 = [0,0,0,0]


ub_xf = [np.pi/2,np.pi/2,0,0]#sys.x_ub # bounds on final state
lb_xf = [np.pi/2,np.pi/2,0,0]

'''
create initial guess

'''

x_guess = np.linspace(ub_x0, ub_xF, ngrid)
u_guess = np.ones([ngrid,sys.m])
t_guess = np.linspace(ub_t0, ub_tf, ngrid)
y_guess= x_guess
dx_guess=np.zeros(x_guess.shape)

for i in range(ngrid):
    dx_guess[i,] = sys.f(x_guess[i,],u_guess[i,]) # compute f

guess_traj = Trajectory(x_guess, u_guess, t_guess, dx_guess, y_guess)

######## set equality contraints (other than dynamics) ##########
# sys.f contains the dynamics ! 

'''
Convert to non-linear program (direct collocation)

decision variables are ordered:
    
    x0[0],x0[1]...,x0[ngrid-1], x1[0],x1[1]...,x1[ngrid-1],...,...xn[0],xn[1]...,xn[ngrid-1]
    u0[0],u0[1]...,u0[ngrid-1], u1[0],x1[1]...,u1[ngrid-1],...,...un[0],un[1]...,un[ngrid-1],
    t0, tf
'''

#convert bounds in discete form, for all x(.),u(.),t(.)
bnds = np.array([]).reshape(0,2) # initialize bounds
for i in range(sys.n): # convert bounds on x
    bnd_arr_to_add = np.concatenate([np.matlib.repmat(lb_x[i], ngrid, 1),np.matlib.repmat(ub_x[i], ngrid, 1)],axis=1)
    bnd_arr_to_add[0,:]=np.array([lb_x0[i],ub_x0[i]])#enforce bound on initial value
    bnd_arr_to_add[ngrid-1,:]=np.array([lb_xf[i],ub_xf[i]])#enforce bound on final value
    bnds = np.append(bnds,bnd_arr_to_add,axis=0)

for i in range(sys.m): # convert bounds on u
    bnd_arr_to_add = np.concatenate([np.matlib.repmat(lb_u[i], ngrid, 1),np.matlib.repmat(ub_u[i], ngrid, 1)],axis=1)
    bnds = np.append(bnds,bnd_arr_to_add,axis=0)

# append bounds on t0 and tF
bnds=np.append(bnds,np.array([lb_t0,lb_t0]).reshape(1,2),axis=0)
bnds=np.append(bnds,np.array([lb_tf,lb_tf]).reshape(1,2),axis=0)

A = bnds[:,1].reshape(ngrid*(sys.n+sys.m)+2,1)

bnds = tuple(map(tuple, bnds))#convert bnds np.array to tuple for input in NLP solver

# equality contraint for dynamics
#for i in range(ngrid):


def unpack_dec_var(decision_variables):
    global ngrid 
    x = np.zeros([ngrid,sys.n])
    u = np.zeros([ngrid,sys.m])
    # unpack decision variables into trajectory
    for i in range(0,sys.n):
        x[:,i] = decision_variables[i*ngrid:(i+1)*ngrid].reshape(ngrid)

    for i in range(sys.n,sys.n+sys.m):
        u[:,i-sys.n] = decision_variables[i*ngrid:(i+1)*ngrid].reshape(ngrid)
    
    t0 = decision_variables[-2]
    tf = decision_variables[-1]
    return x,u,t0,tf

def compute_dx(x,u,sys):
    dx = np.zeros(x.shape)
    global ngrid 
    for i in range(ngrid):
        dx[i,:]=sys.f(x[i,:],u[i,:])
    
    return dx
    

def dynamics_cstr(decision_variables):
    global ngrid        
    x,u,t0,tf = unpack_dec_var(decision_variables)
    h = (tf-t0)/(ngrid-1) #step between each grid
    dx = compute_dx(x,u,sys)
    dyn_constr = np.diff(x,axis=0) - (dx[0:-1]+dx[1:])*h/2 # this must be = 0
    return dyn_constr
    
#dynamics_cstr(A)
#    sys.f(np.array([1,2,3,4]),np.array([3,2])) # compute f

cons = ({'type': 'eq', 'fun': dynamics_cstr(x)})

#dec_var = # decision variables
#optim_traj = Trajectory(x_opt, u_opt, t_opt, dx_opt, y_opt)
#lb_dec_var = ub_x*ones(x.shape)

'''
Solve non-linear program
'''

res4 = minimize(fun3, x0_0, method='SLSQP', bounds=bnds, constraints=cons,tol=1e-6)



'''
Interpolate solution in trajectory object
'''

#sc.interpolate.interp1d(x, y, kind='linear')# linear interpolations
#sc.interpolate.interp1d(x, y, kind='quadratic')# quadratic spline interpolations