# queries
import random
import collections

from utils import *

import numpy as np
import networkx as nx

"""
All the querying strategies are implemented here
"""

def pagerank(bgt, step_size, partial_net, edge_wt):
    """
    PageRank query strategy
    """
    # pagerank is obtained when setting nan as 0.
    cur_A = partial_net['A'].copy()
    A = np.zeros_like(partial_net['A'])
    n = A.shape[0]
    known_eid = partial_net['known_eid']
    known_e = np.array(eid_to_e(n, known_eid))
    A[known_e[:, 0], known_e[:, 1]] = cur_A[known_e[:, 0], known_e[:, 1]]
    A[known_e[:, 1], known_e[:, 0]] = cur_A[known_e[:, 0], known_e[:, 1]]

    G = nx.from_numpy_matrix(np.matrix(A))
    pg = nx.pagerank(G, alpha=0.85)

    utility = []
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)
    for i in range(len(pool_eid)):
        l_e = pool_e[i, :]
        utility.append(pg[l_e[0]]+pg[l_e[1]])

    utility = utility * edge_wt

    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k], reverse=True)
    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return utility, pool_eid[inds]


def degree_sum(bgt, step_size, partial_net, edge_wt):
    """
    Degree sum query strategy
    """
    cur_A = partial_net['A'].copy()
    A = np.zeros_like(partial_net['A'])
    n = A.shape[0]
    known_eid = partial_net['known_eid']
    known_e = eid_to_e(n, known_eid)
    A[known_e[:, 0], known_e[:, 1]] = cur_A[known_e[:, 0], known_e[:, 1]]
    A[known_e[:, 1], known_e[:, 0]] = cur_A[known_e[:, 0], known_e[:, 1]]
    degrees = A.sum(axis=0)

    utility = []
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)
    for i in range(len(pool_eid)):
        l_e = pool_e[i]
        utility.append(degrees[l_e[0]] + degrees[l_e[1]])

    utility = utility * edge_wt

    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k], reverse=True)
    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return utility, pool_eid[inds]


def probability(bgt, step_size, partial_net, post_P, edge_wt):
    """
    Maximum Probability based query strategy
    """
    n = partial_net['A'].shape[0]
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)
    utility = post_P[pool_e[:, 0], pool_e[:, 1]]
    utility = utility * edge_wt
    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k].sum(), reverse=True)
    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return utility, pool_eid[inds]


def distance(bgt, step_size, partial_net, X, edge_wt):
    """
    Minimum distance based query strategy
    """
    n = partial_net['A'].shape[0]
    utility = []
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)
    for i in range(len(pool_eid)):
        l_e = pool_e[i]
        diff = X[l_e[0], :] - X[l_e[1], :]
        utility.append(np.linalg.norm(diff))

    utility = utility * edge_wt

    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k], reverse=False)
    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return utility, pool_eid[inds]


def entropy(bgt, step_size, partial_net, post_P, edge_wt):
    """
    Maximum entropy based query strategy
    """
    A = partial_net['A']
    n = A.shape[0]
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)
    Ps = post_P[pool_e[:, 0], pool_e[:, 1]]

    P_te = Ps
    utility = -P_te*np.log(P_te) - (1-P_te)*np.log(1-P_te)

    utility = utility * edge_wt

    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k], reverse=True)
    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return utility, pool_eid[inds]


def fisher_x_ii_k(partial_net, post_P, X, ne_params, i):
    # sum over only the known neighbours of i
    d = ne_params['d']
    gamma = (1/ne_params['s1']**2 - 1/ne_params['s2']**2)
    known_i = partial_net['known_dic'][i]

    post_P = post_P[i, known_i]
    p = post_P*(1 - post_P)

    tmp = (p*(X[i, :] - X[known_i, :]).T).dot(X[i, :] - X[known_i, :])
    FI_ii = gamma**2*tmp + np.diag(np.ones(d)*np.finfo(float).eps)
    return FI_ii


def d_optimality(bgt, step_size, partial_net, post_P, X, ne_params, edge_wt):
    """
    Parameter Variance reduction strategy
    """
    A = partial_net['A']
    gamma = (1/ne_params['s1']**2 - 1/ne_params['s2']**2)
    n = A.shape[0]
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)

    fishers = [fisher_x_ii_k(partial_net, post_P, X, ne_params, idx) for idx in range(n)]

    utility = []
    for eid in range(len(pool_eid)):
        i, j = pool_e[eid]
        diff_ij = X[i, :] - X[j, :]
        P_ij = post_P[i, j]
        I_ij_X = fishers[i] + fishers[j]

        tmp = diff_ij.dot(np.linalg.inv(I_ij_X)).dot(diff_ij.T)
        res_eid = 2*gamma**2*P_ij*(1-P_ij)*tmp*np.linalg.det(I_ij_X)
        utility.append(res_eid)

    utility = utility * edge_wt

    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k], reverse=True)
    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return np.array(utility), pool_eid[inds]


def v_optimality_k(bgt, step_size, partial_net, post_P, X, ne_params, edge_wt):
    """
    Prediction Variance reduction strategy
    """
    A = partial_net['A']
    gamma = (1/ne_params['s1']**2 - 1/ne_params['s2']**2)
    n = A.shape[0]

    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)
    fishers = [fisher_x_ii_k(partial_net, post_P, X, ne_params, idx) for idx in range(n)]

    l_target_e = partial_net['target_e'].tolist()
    target_dict = collections.defaultdict(list)
    for u, v in l_target_e:
        target_dict[u].append(v)
        target_dict[v].append(u)

    Ps = post_P[pool_e[:, 0], pool_e[:, 1]]
    Ps = Ps*(1-Ps)

    djj_i = [(X[i, :] - X[j, :]).dot(np.linalg.inv(fishers[i])).dot(X[i, :] - X[j, :]) for i, j in pool_e]
    dii_j = [(X[i, :] - X[j, :]).dot(np.linalg.inv(fishers[j])).dot(X[i, :] - X[j, :]) for i, j in pool_e]

    deno_i = 1/(1+gamma**2*Ps*djj_i)
    deno_j = 1/(1+gamma**2*Ps*dii_j)

    utility = []
    for eid in range(len(pool_eid)):
        i, j = pool_e[eid]
        X_diff_ij = X[i, :] - X[j, :]

        Pis = post_P[i, target_dict[i]]
        dkj_i = (X[i, :] - X[target_dict[i], :]).dot(np.linalg.inv(fishers[i])).dot(X_diff_ij.T)
        score_i = ((Pis*(1-Pis))**2*dkj_i**2*deno_i[eid])

        Pjs = post_P[j, target_dict[j]]
        dli_j = (X[j, :] - X[target_dict[j], :]).dot(np.linalg.inv(fishers[j])).dot(-X_diff_ij.T)
        score_j = ((Pjs*(1-Pjs))**2*dli_j**2*deno_j[eid])

        res_eid = gamma**4*Ps[eid]*(np.sum(score_i) + np.sum(score_j))
        utility.append(res_eid)

    utility = utility * edge_wt

    sorted_e_ind = sorted(range(len(utility)), key=lambda k: utility[k], reverse=True)

    if bgt >= step_size:
        inds = sorted_e_ind[:step_size]
    else:
        inds = sorted_e_ind[:min(bgt, len(pool_eid))]
    return np.array(utility), pool_eid[inds]

def distance_weight(partial_net, X):
    """
    Information Density technique with 
    cosine-similarity distance weighting. 
    """
    A = partial_net['A']
    n = A.shape[0]
    pool_eid = partial_net['pool_eid']
    pool_e = eid_to_e(n, pool_eid)

    X_norm_sq = np.linalg.norm(X, axis = 1)**2

    node1_embed = X[pool_e[:,0],:]
    node2_embed = X[pool_e[:,1],:]

    node11_dot = node1_embed @ node1_embed.T
    node12_dot = node1_embed @ node2_embed.T
    node22_dot = node2_embed @ node2_embed.T

    dir1_dot = node11_dot + node22_dot
    dir2_dot = node12_dot + node12_dot.T

    assert(dir1_dot.shape == (len(pool_eid), len(pool_eid)))
    assert(dir2_dot.shape == (len(pool_eid), len(pool_eid)))

    dist = np.minimum(dir1_dot, dir2_dot)

    edge_norms = X_norm_sq[pool_e[:,0]] + X_norm_sq[pool_e[:,1]]
    edge_norms = np.sqrt(edge_norms)
    norm_prod = edge_norms.T @ edge_norms

    cos_dist = dist / norm_prod

    assert(cos_dist.shape == (len(pool_eid), len(pool_eid)))

    dist_weight = np.mean(cos_dist, axis = 1)
    return dist_weight


def get_query(partial_net, post_P, X, step_size, bgt, ne_params, strategy, use_dist_wt = False):
    print('pool_eid length left', len(partial_net['pool_eid']))
    if strategy == 'random_1' or strategy == 'random_2' or strategy == 'random_3':
        pool_eid = partial_net['pool_eid']
        set_pool = set(pool_eid)
        if bgt >= step_size:
            query = np.array(random.sample(list(set_pool), step_size))
        else:
            query = np.array(random.sample(list(set_pool), min(bgt, len(pool_eid))))
    else:
        if use_dist_wt:
            wts = distance_weight(partial_net, X)
        else:
            wts = np.ones(len(partial_net['pool_eid']))
        if strategy == 'pagerank':
            utility, query = pagerank(bgt, step_size, partial_net, wts)
        elif strategy == 'max_degree_sum':
            utility, query = degree_sum(bgt, step_size, partial_net, wts)
        elif strategy == 'max_probability':
            utility, query = probability(bgt, step_size, partial_net, post_P, wts)
        elif strategy == 'min_distance':
            utility, query = distance(bgt, step_size, partial_net, X, wts)
        elif strategy == 'max_entropy':
            utility, query = entropy(bgt, step_size, partial_net, post_P, wts)
        elif strategy == 'd_optimality':
            utility, query = d_optimality(bgt, step_size, partial_net, post_P, X, ne_params, wts)
        elif strategy == 'v_optimality':
            utility, query = v_optimality_k(bgt, step_size, partial_net, post_P, X, ne_params, wts)
        else:
            query = []
    return query