import numpy as np

from .system import ContinuousDynamicSystem

class StateSpaceSystem(ContinuousDynamicSystem):
    """Time-invariant state space representation of dynamic system

    ```
    f = A*x + B*u
    h = C*x + D*u
    ```

    Parameters
    ----------
    A, B, C, D : array_like
        The matrices which define the system

    """

    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self._check_dimensions()

        n = A.shape[1]
        m = B.shape[1]
        p = C.shape[0]

        super().__init__(n, m, p)

    def _check_dimensions(self):
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")

        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("Number of rows in B does not match A")

        if self.C.shape[1] != self.A.shape[0]:
            raise ValueError("Number of columns in C does not match A")

        if self.D.shape[1] != self.B.shape[1]:
            raise ValueError("Number of columns in D does not match B")

        if self.C.shape[0] != self.D.shape[0]:
            raise ValueError("Number of rows in C does not match D")

    def f(self, x, u, t):
        return np.dot(self.A, x) + np.dot(self.B, u)

    def h(self, x, u, t):
        return np.dot(self.C, x) + np.dot(self.D, u)


def _approx_jacobian(f, x0, epsilons):
    """Numerically approximate the jacobian of a function

    Parameters
    ----------
    f : callable
        Function for which to approximate the jacobian. Must accept an array of
        dimension ``n`` and return an array of dimension ``m``.
    x0 : array_like (dimension ``n``)
        Input around which the jacobian will be evaluated.
    epsilons : array_like (dimension ``n``)
        Step size to use for each input when approximating the jacobian

    Returns
    -------
    jac : array_like
        Jacobian matrix with dimensions m x n
    """

    n = x0.shape[0]
    m = f(x0).shape[0]

    jac = np.zeros((m, n))

    for j in range(n):
        xf = np.copy(x0)
        xf[n] = xf[n] + epsilons[n]

        xb = np.copy(x0)
        xb[n] = xb[n] - epsilons[n]

        jac[:, n] = (f(xf) - f(xb)) / (2 * epsilons[n])

    return jac

def linearize(sys, x0, u0, epsilon_x, epsilon_u=None):
    """Generate linear state-space model by linearizing any system.

    The system to be linearized is assumed to be time-invariant.

    Parameters
    ----------
    sys : `pyro.dynamic.ContinuousDynamicSystem`
        The system to linearize
    x0 : array_like
        State array arround which the system will be linearized
    epsilon : float
        Step size to use for numerical gradient approximation

    Returns
    -------
    instance of `StateSpaceSystem`

    """

    epsilon_x = np.asarray(epsilon_x)

    if epsilon_u is None:
        if epsilon_x.size > 1:
            raise ValueError("If epsilon_u is not provided, epsilon_x must be scalar")
        epsilon_u = epsilon_x

    epsilon_u = np.asarray(epsilon_u)

    if epsilon_u.size == 1:
        epsilon_u = np.ones(sys.m) * epsilon_u

    if epsilon_x.size == 1:
        epsilon_x = np.ones(sys.n) * epsilon_x

    def f_x(x):
        return sys.f(x, u0, 0)

    def f_u(u):
        return sys.f(x0, u, 0)

    def h_x(x):
        return sys.h(x, u0, 0)

    def h_u(u):
        return sys.h(x0, u, 0)

    A = _approx_jacobian(f_x, x0, epsilon_x)
    B = _approx_jacobian(f_u, u0, epsilon_u)
    C = _approx_jacobian(h_x, x0, epsilon_x)
    D = _approx_jacobian(h_u, u0, epsilon_u)

    return StateSpaceSystem(A, B, C, D)