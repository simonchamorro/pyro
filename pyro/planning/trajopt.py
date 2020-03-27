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
'''
################################################################################
'''
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
sys = manipulator.TwoLinkManipulator()
sys.f(np.array([1,2,3,4]),np.array([3,2])) # compute f
sys.x0 =np.array([0.1,0.1,0,0])
tf=4
sys.compute_trajectory( tf )

#ctl  = plan.OpenLoopController.load_from_file( 'double_pendulum_rrt.npy' )
ctl = plan.OpenLoopController(sys.traj)
ctl.trajectory.u=np.ones(sys.traj.u.shape)*10
# New cl-dynamic
cl_sys = ctl + sys

#sys.traj.u=np.ones(sys.traj.t.shape)*10
cl_sys.compute_trajectory( tf )
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()
