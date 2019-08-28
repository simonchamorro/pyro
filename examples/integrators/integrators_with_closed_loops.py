# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:24:51 2018

@author: Alexandre
"""

############################################################################
from pyro.dynamic  import integrator
from pyro.control  import linear
############################################################################

# Double integrator
si = integrator.SimpleIntegrator()
di = integrator.DoubleIntegrator()
ti = integrator.TripleIntegrator()

# Controller 
ctl      = linear.ProportionnalSingleVariableController()
ctl.gain = 2

# New cl-dynamic
clsi = ctl + si
clsi_sim = clsi.compute_trajectory(x0=[10], tf=10)
clsi.plot_phase_plane_trajectory(clsi_sim, 0, 0)
clsi.plot_trajectory(clsi_sim, 'xu')

cldi = ctl + di
cldi_sim = cldi.compute_trajectory(x0=[10, 0], tf=10)
cldi.plot_phase_plane_trajectory(cldi_sim)
cldi.plot_trajectory(cldi_sim, 'xu')

clti = ctl + ti

clti_sim = clti.compute_trajectory(x0=[0.1,0,0], tf=10)
clti.plot_phase_plane_trajectory_3d(clti_sim)
clti.plot_trajectory(clti_sim, 'xu')