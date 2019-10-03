
import numpy as np

from scipy import integrate

from pyro.control.linear import PIDController

from pyro.dynamic.system import ContinuousDynamicSystem

class FirstOrder(ContinuousDynamicSystem):
    def __init__(self, tau):
        super().__init__(1, 1, 1)
        self.tau = tau

    def f(self, x, u, t):
        return (u - x) / self.tau

def step(a, delay=0):
    def u(t):
        if t < delay:
            return 0
        else:
            return a
    return u

def _deriv(y, x):
    dy = np.empty(y.shape)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    for i in range(1, (dy.shape[0] - 1)):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy

def test_sdof_prop():
    """Test single DOF system with proportional control"""
    tau_p = 2
    tf = 1.5
    kp = 20

    sys = FirstOrder(tau_p)
    ctl = PIDController([[kp]], [[0]], [[0]])
    clsys = ctl + sys

    sim = clsys.compute_trajectory(x0_sys=[0], r=step(1), tf=tf, n=100)
    sys.plot_trajectory(sim, 'xu')

    # analytic solution
    kcl = kp / (kp + 1) # steady state asymptote value
    tau_cl = tau_p / (kp + 1) # Closed loop time constant
    x_ref = kcl * (1-np.exp(-sim.t/tau_cl))

    assert np.allclose(sim.x[:, 0], x_ref)

def test_sdof_pid():
    """Check PID controller outputs"""
    sys = FirstOrder(1)
    ctl = PIDController([[2]], [[1]], [[0]], dv_tau=0.001)
    clsys = ctl + sys

    sim = clsys.compute_trajectory(x0_sys=0, r=step(1), tf=2, n=500)
    #sys.plot_trajectory(sim, 'xu')

    # error
    e = (sim.r - sim.y)[:, 0]

    # Check that integrator values are OK
    e_int = integrate.cumtrapz(e, sim.t, initial=0)
    assert np.allclose(sim.x[:, 1], e_int, atol=1e-8, rtol=1E-3)

    # Check controller output
    ctl_ref = (e * ctl.KP + e_int * ctl.KI).reshape((sim.n,))
    assert np.allclose(sim.u[:, 0], ctl_ref, atol=0, rtol=1E-4)

    numder = _deriv(e, sim.t)
    ctlder = (e - sim.x[:, 2]) / ctl.dv_tau

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(sim.t, e, label="error")
    plt.plot(sim.t, ctlder, '.', label="ctl filtered deriv")
    plt.plot(sim.t, numder, label="numerical error deriv")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_sdof_pid()