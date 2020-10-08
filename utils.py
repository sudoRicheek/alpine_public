# utility functions

import os
import sys

import numpy as np
import collections
import pickle
import random
import scipy.io
import scipy.sparse as sparse
from sklearn.metrics import roc_auc_score

from cne import maxent
from cne.cne_known import ConditionalNetworkEmbedding_K


def from_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None


def to_cache(cache_file, data):
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def memoize(func, cache_file, refresh=False):
    def memoized_func(*args):
        result = from_cache(cache_file)
        if result is None or refresh is True:
            result = func(*args)
            to_cache(cache_file, result)
        return result
    return memoized_func


def eid_to_e(n, eid):
    # edge list is np.array
    col = eid%n
    row = eid//n
    return np.vstack([row, col]).T


def e_to_eid(n, e):
    # eids is np.array
    return n*np.array(e)[:, 0] + np.array(e)[:, 1]


def from_csr_matrix_to_edgelist(csr_A):
    csr_A = sparse.triu(csr_A, 1).tocsr()
    t_list = csr_A.indices
    h_list = np.zeros_like(t_list).astype(int)
    for i in range(csr_A.shape[0]):
        h_list[csr_A.indptr[i]:csr_A.indptr[i+1]] = i
    return np.vstack((h_list, t_list)).T


def generate_S0(A, S0_r):
    # for Case-1&2 with T=U
    n = A.shape[0]
    m = 0.5*n*(n-1)
    u_eid = e_to_eid(n, from_csr_matrix_to_edgelist(sparse.triu(np.ones_like(A), 1)))
    nr_S0 = int(m*S0_r)
    print('nr_S0', nr_S0)

    i_start = random.randint(0, n-1)
    dfst_A = sparse.csgraph.depth_first_tree(A, i_start, directed=False)
    dfst_A = (dfst_A + dfst_A.T).astype(bool)
    dfst_e = from_csr_matrix_to_edgelist(dfst_A)
    dfst_eid = e_to_eid(n, dfst_e)
    print('dfst_e.shape[0]', dfst_e.shape[0])
    nr_S0 -= dfst_e.shape[0]

    set_pool = set(u_eid) - set(dfst_eid)
    rest_S0_eid = np.array(random.sample(list(set_pool), nr_S0))

    S0_eid = np.hstack((dfst_eid, rest_S0_eid))
    return S0_eid


def split_node_pairs(A, S0_r, target_size):
    # for Case-3
    n = A.shape[0]
    m = 0.5*n*(n-1)
    all_eid = e_to_eid(n, from_csr_matrix_to_edgelist(sparse.triu(np.ones_like(A), 1)))
    nr_S0 = int(m*S0_r)
    print('nr_S0', nr_S0)

    i_start = random.randint(0, n-1)
    dfst_A = sparse.csgraph.depth_first_tree(A, i_start, directed=False)
    dfst_A = (dfst_A + dfst_A.T).astype(bool)
    dfst_e = from_csr_matrix_to_edgelist(dfst_A)
    dfst_eid = e_to_eid(n, dfst_e)
    print('dfst_e.shape[0]', dfst_e.shape[0])
    nr_S0 -= dfst_e.shape[0]

    # assume there are at least 10 links in the test set.
    pool_pos_A = sparse.triu(A.astype(int) - dfst_A.astype(int), 1) > 0
    pool_pos_e = from_csr_matrix_to_edgelist(pool_pos_A)
    pool_pos_eid = e_to_eid(n, pool_pos_e)
    rand_inds = list(range(len(pool_pos_eid)))
    random.shuffle(rand_inds)
    target_pos_0 = rand_inds[:10]
    target_eid_0 = pool_pos_eid[target_pos_0]

    set_pool = set(all_eid) - set(dfst_eid) - set(target_eid_0)
    rest_S0_eid = np.array(random.sample(list(set_pool), nr_S0))
    S0_eid = np.hstack((dfst_eid, rest_S0_eid))

    set_pool = set(all_eid) - set(S0_eid) - set(target_eid_0)
    rest_target_eid = np.array(random.sample(list(set_pool), target_size-10))
    target_eid = np.hstack((target_eid_0, rest_target_eid))

    return S0_eid, target_eid


def get_partial_net(full_A, known_eid, unknown_eid, pool_eid=None, target_eid=None):
    n = full_A.shape[0]
    partial_A = full_A.copy()
    if pool_eid is None:
        pool_eid = unknown_eid.copy()
    if target_eid is None:
        target_eid = unknown_eid.copy()
    target_e = eid_to_e(n, target_eid)

    known_e = eid_to_e(n, known_eid)
    l_known_e = known_e.tolist()
    known_dict = collections.defaultdict(list)
    for u, v in l_known_e:
        known_dict[u].append(v)
        known_dict[v].append(u)

    partial_network = {'A': partial_A,
                       'known_eid': known_eid,
                       'known_dic': known_dict,
                       'u_eid': unknown_eid,
                       'pool_eid': pool_eid,
                       'target_eid': target_eid,
                       'target_e': target_e,
                       }
    return partial_network


def update_partial_net(partial_net, query, full_A):
    n = full_A.shape[0]
    known_dic = partial_net['known_dic'].copy()
    query_e = eid_to_e(n, query)
    l_query_e = query_e.tolist()
    for u, v in l_query_e:
        known_dic[u].append(v)
        known_dic[v].append(u)

    new_net = {'A': partial_net['A'],
               'known_eid': np.hstack((partial_net['known_eid'], query)),
               'known_dic': known_dic,
               'u_eid': np.setdiff1d(partial_net['u_eid'], query),
               'pool_eid': np.setdiff1d(partial_net['pool_eid'], query),
               'target_eid': partial_net['target_eid'],
               'target_e': partial_net['target_e'],
               }
    return new_net


def embed(partial_net, X0, ne_params):
    cur_A = partial_net['A'].copy()
    n = cur_A.shape[0]
    known_dic = partial_net['known_dic']

    te_A = np.zeros_like(partial_net['A'])
    known_eid = partial_net['known_eid']
    known_e = eid_to_e(n, known_eid)
    te_A[known_e[:, 0], known_e[:, 1]] = cur_A[known_e[:, 0], known_e[:, 1]]
    te_A[known_e[:, 1], known_e[:, 0]] = cur_A[known_e[:, 0], known_e[:, 1]]

    # degree prior after Laplace smoothing
    unknown_eid = partial_net['u_eid']
    unknowns_e = eid_to_e(n, unknown_eid)
    A_temp = te_A.copy()
    if len(unknowns_e) != 0:
        N = 0.01
        f = np.sum(te_A)/(n*(n-1))
        A_temp = (A_temp-1)*(-f*N)/(1+N) + A_temp*(1+f*N)/(1+N)
        A_temp[unknowns_e[:, 0], unknowns_e[:, 1]] = f
        A_temp[unknowns_e[:, 1], unknowns_e[:, 0]] = f
        A_temp -= np.diag(np.diag(A_temp))
    prior = maxent.BGDistr(A_temp, datasource='custom')
    prior.fit(undirected=True, iterations=100, method='L-BFGS-B', verbose=False)

    # Use CNE to fit only the known part
    cne_model = ConditionalNetworkEmbedding_K(A_temp, ne_params, known_e, known_dic, partial_net, prior=prior)
    cne_model.fit(ftol=1e-4, verbose=False, X0=X0)

    X = cne_model.get_embedding()
    post_P = np.array([cne_model.compute_row_posterior(row_i, range(n)) for row_i in range(n)])

    return X, post_P


def predict(post_P, e):
    return post_P[e[:, 0], e[:, 1]]


def eval_prediction(y_true, y_pred, type):
    if type == 0:
        s = np.sum(np.log((-1)**y_true * (1 - y_true - y_pred)))
    elif type == 1:
        s = roc_auc_score(y_true, y_pred)
    else:
        print('No such evaluation criterion.')
    return s


def load_data(dataname):
    if dataname == 'polbooks':
        full_A = scipy.io.loadmat('./dataset/polbooks.mat')
        full_A = full_A['Problem'][0]['A'][0]
        full_A = np.array(full_A.todense()).astype(bool).astype(float)
    elif dataname == 'celegans':
        full_A = scipy.io.loadmat('./dataset/Celegans.mat')
        full_A = np.array(full_A['net'].todense()).astype(bool).astype(float)
    elif dataname == 'usair':
        full_A = scipy.io.loadmat('./dataset/USAir.mat')
        full_A = np.array(full_A['net'].todense()).astype(bool).astype(float)
    elif dataname == 'polblogs_cc':
        full_A = from_cache('./dataset/polblogs_cc.pkl')
        full_A = np.array(full_A.todense())
    elif dataname == 'mp_cc':
        full_A = from_cache('./dataset/twitter_mp_cc.pkl')
        full_A = np.array(full_A.todense())
    elif dataname == 'ppi_cc':
        full_A = from_cache('./dataset/ppi_cc.pkl')
        full_A = np.array(full_A.todense())
    elif dataname == 'blog':
        full_A = scipy.io.loadmat('./dataset/blog.mat')
        full_A = np.array(full_A['network'].todense()).astype(bool).astype(float)
    else:
        print('No such dataset!')
    full_A = (full_A + full_A.T).astype(bool).astype(float)
    return full_A


def strategy_collections():
    strategy = ['random_1', 'random_2', 'random_3',
                'pagerank', 'max_degree_sum',
                'max_probability', 'min_distance',
                'max_entropy',
                'd_optimality', 'v_optimality'
                ]
    labels = ['rand.',
              'page_rank.', 'max_deg_s.',
              'max-prob.', 'min-dis.',
              'max-ent.',
              'd-opt.', 'v-opt.'
              ]
    return strategy, labels
