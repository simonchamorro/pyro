
import numpy as np

from scipy import integrate, signal

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

def filtered_deriv(y, x, tau=0):
    """Numerical derivative with optional lowpass filter.

    tau is the filter time constant expressed in same units as x (eg seconds if x is
    time).
    """
    dy = np.empty(y.shape)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    for i in range(1, (dy.shape[0] - 1)):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    # No filter
    if tau <= 0:
        return dy

    # sample freq
    fs = x.shape[0] / (x[-1] - x[0])
    nyqfreq = fs / 2
    w0 = 1 / (tau*2*np.pi*nyqfreq)

    lowpass = signal.iirfilter(1, w0, btype='lowpass', analog=False)

    # Create initial conditions so y=0 at t=0
    zi = signal.lfiltic(*lowpass, y=[0], x=dy[:1])
    print(zi)

    filtered, _ = signal.lfilter(*lowpass, dy, zi=-zi)
    return filtered


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
    tf = 2 # simulation time
    npts = 500 # simulation samples
    tau = 1E-2 # time constant of derivative filter

    # Create system/controller and run simulation
    sys = FirstOrder(1)
    ctl = PIDController([[3]], [[2]], [[0.6]], dv_tau=tau)
    clsys = ctl + sys
    sim = clsys.compute_trajectory(x0_sys=0, r=step(1), tf=tf, n=npts)

    # error
    e = (sim.r - sim.y)[:, 0]

    # integral of error
    e_int = integrate.cumtrapz(e, sim.t, initial=0)

    # filtered derivative of error
    e_fd = filtered_deriv(e, sim.t, tau)

    # Check controller output
    ctl_ref = (e * ctl.KP + e_int * ctl.KI + e_fd * ctl.KD).reshape((sim.n,))

    assert np.allclose(sim.u[:, 0], ctl_ref, atol=0, rtol=1E-2)

if __name__ == "__main__":
    pass