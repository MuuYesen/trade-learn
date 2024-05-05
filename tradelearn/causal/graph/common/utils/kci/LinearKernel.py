from __future__ import annotations

from numpy import ndarray

from tradelearn.causal.graph.common.utils.kci.Kernel import Kernel


class LinearKernel(Kernel):
    def __init__(self):
        Kernel.__init__(self)

    def kernel(self, X: ndarray, Y: ndarray | None = None):
        """
        Computes the linear kernel k(x,y)=x^Ty
        """
        if Y is None:
            Y = X
        return X.dot(Y.T)
