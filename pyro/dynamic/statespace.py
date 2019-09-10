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