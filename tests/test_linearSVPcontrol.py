
import numpy as np

from matplotlib import pyplot as plt

import pytest

from pyro.dynamic import integrator
from pyro.control import linear

@pytest.fixture
def controlled_point_mass():
    """DoubleInteg with proportional velocity control"""

    # Dynamic system
    pm = integrator.DoubleIntegrator()
    pm.ubar = np.array([0,])

    def get_velocity(x, u, t):
        return x[1]

    pm.h = get_velocity

    # Controller
    ctl = linear.ProportionnalSingleVariableController()
    ctl.gain = 20

    cls = ctl + pm
    return cls

def test_svpc_step(controlled_point_mass):
    """Test SVPC with integrator and step input"""
    cls = controlled_point_mass

    x0 = [0, 3]
    cls.compute_trajectory(x0)
    sim = cls.sim

    # Find index corresponding to t = 2 seconds
    t_index = np.where(sim.t >= 2)[0][0]
    zeros = np.zeros(sim.t[t_index:].shape)

    # Plot simulation
    #cls.sim.plot('xu')
    #plt.ioff(); plt.show()

    # Check that we initially have an error vs the setpoint
    assert(sim.x_sol[0, 1] - cls.ctl.rbar == x0[1])

    # Check that controller output == gain*error
    expected_control_u = -cls.ctl.gain * (sim.x_sol[:, 1] - cls.ctl.rbar)[:, np.newaxis]
    assert(np.allclose(sim.u_sol, expected_control_u))

    # Check that we reach stable 0 speed after 2 seconds
    assert(np.allclose(sim.x_sol[t_index:, 1], zeros))

    # Check that controller output is 0 after 2 seconds
    assert(np.allclose(sim.u_sol[t_index:], zeros))