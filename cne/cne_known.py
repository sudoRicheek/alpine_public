import numpy as np

from collections import defaultdict

"""
This is the implementation of the CNE-Known model
"""

class ConditionalNetworkEmbedding_K:
    def __init__(self, A, ne_params, known_e, known_dic, partial_net, prior=None):
        self.__A = A
        self.__d = ne_params['d']
        self.__s1 = ne_params['s1']
        self.__s2 = ne_params['s2']
        self.__lr = ne_params['optimizer']['lr']
        self.__max_iter = ne_params['optimizer']['max_iter']
        self.__s_diff = (1/ne_params['s1']**2 - 1/ne_params['s2']**2)
        self.__s_div = ne_params['s1']/ne_params['s2']
        self.__prior = prior
        self.__known_dic = known_dic

    def _obj_grad(self, X, A, prior, s_div, s_diff):
        """
        Calculate the gradient for maximum likelihood estimation
        keeping the knowns in consideration.
        """
        res_obj = 0.
        res_grad = np.zeros_like(X)
        n = X.shape[0]
        for xid in range(n):
            knowns = self.__known_dic[xid]
            A_i_k = np.array(A[xid, knowns])
            prior_i = prior.get_row_probability(xid, knowns)
            diff = (X[xid, :] - X[knowns, :]).T
            d_p2 = np.sum(diff**2, axis=0)

            post = self._posterior(1, d_p2, prior_i, s_div, s_diff)
            prob = (2*post-1)*A_i_k + 1 - post
            obj = np.log(prob+1e-20)
            res_obj += np.sum(obj)

            grad = s_diff*((post-A_i_k)*diff).T
            res_grad[xid, :] += np.sum(grad, axis=0)
            res_grad[knowns, :] -= grad
        return -res_obj, -res_grad

    def _row_posterior(self, row_id, col_ids, X, prior, s_div, s_diff):
        prior = prior.get_row_probability(row_id, col_ids)
        d_p2 = np.sum(((X[row_id, :] - X[col_ids, :]).T)**2, axis=0)
        return self._posterior(1, d_p2, prior, s_div, s_diff)

    def _posterior(self, obs_val, d_p2, prior, s_div, s_diff):
        if obs_val == 1:
            return 1./(1+(1-prior)/prior*s_div*np.exp(d_p2/2*s_diff))
        else:
            return 1./(1+prior/(1-prior)/s_div*np.exp(-d_p2/2*s_diff))

    def _optimizer_adam(self, X, A, prior_dist, s_div, s_diff, num_epochs=2000,
                        alpha=0.2, beta_1=0.9, beta_2=0.9999, eps=1e-8,
                        ftol=1e-7, verbose=False):
        m_prev = np.zeros_like(X)
        v_prev = np.zeros_like(X)
        obj_old = 0.
        for epoch in range(num_epochs):
            obj, grad = self._obj_grad(X, A, prior_dist, s_div, s_diff,)

            # Adam optimizer
            m = beta_1*m_prev + (1-beta_1)*grad
            v = beta_2*v_prev + (1-beta_2)*grad**2

            m_prev = m.copy()
            v_prev = v.copy()

            m = m/(1-beta_1**(epoch+1))
            v = v/(1-beta_2**(epoch+1))
            X -= alpha*m/(v**.5 + eps)

            grad_norm = np.sum(grad**2)**.5
            obj_diff = np.abs(obj_old - obj)

            if verbose:
                print('Epoch: {:d}, grad norm: {:.4f}, obj: {:.4f}, obj diff: {:.4f}'.format(epoch, grad_norm, obj, obj_diff), flush=True)
            if obj_diff/max(obj, obj_old, 1) < ftol:
                print('Epoch: {:d}, grad norm: {:.4f}, obj: {:.4f}, obj diff: {:.4f}'.format(epoch, grad_norm, obj, obj_diff), flush=True)
                break

            obj_old = obj

        return X

    def fit(self, ftol=1e-7, verbose=False, X0=None):
        if X0 is None:
            X = np.random.randn(self.__A.shape[0], self.__d)
        else:
            X = X0.copy()
        self.__emb = self._optimizer_adam(X, self.__A, self.__prior,
                                          self.__s_div, self.__s_diff,
                                          alpha=self.__lr,
                                          num_epochs=self.__max_iter,
                                          ftol=ftol, verbose=verbose)

    def predict(self, E):
        edge_dict = defaultdict(list)
        ids_dict = defaultdict(list)
        for i, edge in enumerate(E):
            edge_dict[edge[0]].append(edge[1])
            ids_dict[edge[0]].append(i)

        pred = []
        ids = []
        for u in edge_dict.keys():
            pred.extend(self._row_posterior(u, edge_dict[u], self.__emb, self.__prior, self.__s_div, self.__s_diff))
            ids.extend(ids_dict[u])

        return [p for _, p in sorted(zip(ids, pred))]

    def get_embedding(self):
        return self.__emb

    def compute_row_posterior(self, row_id, col_ids):
        return self._row_posterior(row_id, col_ids, self.__emb, self.__prior, self.__s_div, self.__s_diff)
