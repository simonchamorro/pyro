import numpy as np

from pyro.dynamic import ContinuousDynamicSystem


###############################################################################
class StateSpaceSystem(ContinuousDynamicSystem):
    """Time-invariant state space representation of dynamic system

    f = A x + B u
    h = C x + D u

    Parameters
    ----------
    A, B, C, D : array_like
        The matrices which define the system

    """
    ############################################
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self._check_dimensions()

        n = A.shape[1]
        m = B.shape[1]
        p = C.shape[0]
        
        ContinuousDynamicSystem.__init__(self, n, m, p)
        
    ############################################
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
    
    #############################################
    def f(self, x, u, t):

        dx = np.dot(self.A, x) + np.dot(self.B, u)

        return dx
    
    #############################################
    def h(self, x, u, t):
        
        y = np.dot(self.C, x) + np.dot(self.D, u)
        
        return y
    
    

################################################################
def _approx_jacobian(func, xbar, epsilons):
    """ Numerically approximate the jacobian of a function

    Parameters
    ----------
    func : callable
        Function for which to approximate the jacobian. Must accept an array of
        dimension ``n`` and return an array of dimension ``m``.
    xbar : array_like (dimension ``n``)
        Input around which the jacobian will be evaluated.
    epsilons : array_like (dimension ``n``)
        Step size to use for each input when approximating the jacobian

    Returns
    -------
    jac : array_like
        Jacobian matrix with dimensions m x n
    """

    n  = xbar.shape[0]
    ybar = func(xbar)
    m  = ybar.shape[0]

    J = np.zeros((m, n))
    
    for j in range(n):
        # Forward evaluation
        xf = np.copy(xbar)
        xf[j] = xf[j] + epsilons[j]

        # Backward evaluation
        xb = np.copy(xbar)
        xb[j] = xb[j] - epsilons[j]
        
        delta = func(xf) - func(xb)

        J[:, j] = delta / (2 * epsilons[j])

    return J


#################################################################
def linearize(sys, epsilon_x, epsilon_u=None):
    """Generate linear state-space model by linearizing any system.

    The system to be linearized is assumed to be time-invariant.

    Parameters
    ----------
    sys : `pyro.dynamic.ContinuousDynamicSystem`
        The system to linearize
    xbar : array_like
        State array arround which the system will be linearized
    epsilon : float
        Step size to use for numerical gradient approximation

    Returns
    -------
    instance of `StateSpaceSystem`

    """
    
    xbar = sys.xbar
    ubar = sys.ubar

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
        return sys.f(x, ubar, 0)

    def f_u(u):
        return sys.f(xbar, u, 0)

    def h_x(x):
        return sys.h(x, ubar, 0)

    def h_u(u):
        return sys.h(xbar, u, 0)

    A = _approx_jacobian(f_x, xbar, epsilon_x)
    B = _approx_jacobian(f_u, ubar, epsilon_u)
    C = _approx_jacobian(h_x, xbar, epsilon_x)
    D = _approx_jacobian(h_u, ubar, epsilon_u)

    return StateSpaceSystem(A, B, C, D)


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    from pyro.dynamic import pendulum
    
    non_linear_sys = pendulum.SinglePendulum()
    
    
    non_linear_sys.xbar = np.array([0,0])
    
    EPS = 0.1 
    
    linearized_sys = linearize( non_linear_sys , EPS )
    
    print(linearized_sys.C)
