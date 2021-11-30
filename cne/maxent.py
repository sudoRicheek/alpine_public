from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import scipy.sparse as sparse
from scipy.optimize import minimize

class BGDistr(object):
    """
    Stores the prior distribution
    """

    def __init__(self, *args, **kwargs):
        super(BGDistr, self).__init__()
        A = sparse.csr_matrix(args[0]).todense()
        A -= np.diag(np.diag(A))
        A = sparse.csr_matrix(A)

        self.data = A
        self.row_sums = []
        self.n_nodes = self.data.shape[0]

        for ridx, row in enumerate(self.data):
            row = row.toarray()
            self.row_sums.append(np.sum(row))

    def lagrangian(self, x):
        E = np.exp(x/2 + x[:,None]/2)
        M = np.log(1 + E)
        M -= np.diag(np.diag(M))
        return np.sum(M) - np.sum(x*self.row_sums)

    def gradient(self, x):
        E = np.exp(x/2 + x[:,None]/2)
        M = E / (1 + E)
        M -= np.diag(np.diag(M))
        return np.sum(M, axis=0) - self.row_sums


    def fit(self, verbose=True, tol=1e-5, iterations=100, **kwargs):
        x0 = np.random.randn(self.n_nodes)
        res = minimize(self.lagrangian, x0,
                       method='L-BFGS-B',
                       jac=self.gradient,
                       options={'disp': verbose, 'maxcor': 20, 'ftol': 1e-40,
                                'gtol': tol, 'maxiter': iterations, 'maxls': 20})
        self.la = res.x

    def get_row_probability(self, row_id, col_ids):
        '''
        Compute prior (degree) probability for the entries in a row specified
        by row_id.
        '''
        row_la = self.la[row_id]
        col_las = self.la[col_ids]

        E = np.exp(row_la/2 + col_las/2)
        P_i = E/(1+E)

        if row_id in col_ids:
            P_i[col_ids.index(row_id)] = 0 + sys.float_info.epsilon

        return P_i
