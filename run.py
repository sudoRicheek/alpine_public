import collections
import math
import os
import random
import sys
from functools import partial
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse as sparse

from cne.cne_known import ConditionalNetworkEmbedding_K
from query import *
from utils import *


def test_for_s(folder_split, partial_net, test_e, y_true, X, post_P, ne_params, step_size, budget, strategy, ne_id, sid):
    s = strategy[sid]
    print('s', s, sid)
    score_s = []

    bgt = budget
    cur_partial_net = partial_net.copy()
    cur_X = X.copy()
    cur_post_P = post_P.copy()
    while bgt > 0:
        print('bgt', bgt)
        y_pred = predict(cur_post_P, test_e)
        score_s.append(eval_prediction(y_true, y_pred, eval_t))
        query = get_query(cur_partial_net, cur_post_P, cur_X, step_size, bgt, ne_params, s)
        print(s, 'len(query)', len(query), query[-10:], score_s[-1])

        cur_partial_net = update_partial_net(cur_partial_net, query, full_A)
        cur_X, cur_post_P = embed(cur_partial_net, None, ne_params)
        bgt -= len(query)

    y_pred = predict(cur_post_P, test_e)
    score_s.append(eval_prediction(y_true, y_pred, eval_t))
    to_cache(folder_split+'/'+s+'_ne_'+str(ne_id)+'.pkl', score_s)
    return score_s


def generate_pool(eids, size):
    return np.array(random.sample(list(eids), size))


def one_split_all_s(p, folder_split, full_A, r_0, stp_s, case, bgt_k, pool_size, target_size, strategy, split_id, nr_ne):
    n = full_A.shape[0]
    all_eid = e_to_eid(n, from_csr_matrix_to_edgelist(sparse.triu(np.ones_like(full_A), 1)))
    if case == 1:
        # Case-1
        S0_eid = memoize(generate_S0, folder_split+'/S0_'+str(split_id)+'.pkl', refresh=False)(full_A, r_0)
        U_eid = np.array(list(set(all_eid) - set(S0_eid)))
        print('S0_eid', len(S0_eid))
        partial_net0 = get_partial_net(full_A, S0_eid, U_eid)
    elif case == 2:
        # Case-2
        S0_eid = memoize(generate_S0, folder_split+'/S0_'+str(split_id)+'.pkl', refresh=False)(full_A, r_0)
        U_eid = np.array(list(set(all_eid) - set(S0_eid)))
        print('S0_eid', len(S0_eid))
        pool_eid = memoize(generate_pool, folder_split+'/pool_'+str(split_id)+'.pkl', refresh=False)(U_eid, pool_size)
        print('pool_eid size', len(pool_eid))
        partial_net0 = get_partial_net(full_A, S0_eid, U_eid, pool_eid)
    elif case == 3:
        # Case-3
        S0_eid, target_eid = memoize(split_node_pairs, folder_split+'/split_'+str(split_id)+'.pkl', refresh=False)(full_A, r_0, target_size)
        U_eid = np.array(list(set(all_eid) - set(S0_eid)))
        print('S0_eid', len(S0_eid))
        print('target_eid', len(target_eid))
        rest_u_eid = np.array(list(set(U_eid) - set(target_eid)))
        pool_eid = memoize(generate_pool, folder_split+'/pool_'+str(split_id)+'.pkl', refresh=False)(rest_u_eid, pool_size)
        print('pool_eid', len(pool_eid))
        partial_net0 = get_partial_net(full_A, S0_eid, U_eid, pool_eid, target_eid)
    else:
        # You can also define your partial net, P, and T here.
        print('No such Case.')

    target_e = partial_net0['target_e']
    y_true = full_A[target_e[:, 0], target_e[:, 1]]

    size_unknown = len(partial_net0['u_eid'])
    print('nr_unknown_e', size_unknown)
    step_size = stp_s
    budget = bgt_k
    print('budget', budget, 'stp_s', stp_s)

    for ne_id in range(nr_ne):
        X0, post_P0 = memoize(embed, folder_split+'/NE_'+str(ne_id)+'.pkl', refresh=False)(partial_net0, None, ne_params)
        res_test_id = p.map(partial(test_for_s, folder_split, partial_net0,
                                    target_e, y_true, X0, post_P0, ne_params,
                                    step_size, budget, strategy, ne_id),
                                    list(range(len(strategy))))
    return 0


if __name__ == "__main__":

    # changing dataset
    dataname = 'celegans'

    # changing parameters for ALPINE experiment
    Case = 2
    r_0 = 0.03
    nr_split = 2
    nr_ne = 2

    # parameters for embedding method
    ne_params = {"name": "cne", "d": 8, "s1": 1, "s2": 32,
                 "optimizer": {"name": "adam", "lr": 0.1, "max_iter": 2000}}

    # load data
    if not os.path.exists(dataname):
        os.makedirs(dataname)

    full_A = load_data(dataname)
    strategy, labels = strategy_collections()
    n = full_A.shape[0]
    m = 0.5*n*(n-1)
    eval_t = 1
    stp_s = int(m*0.01)


    # config active learning setting
    if Case == 1:
        bgt_k = stp_s*10  # changing parameter
        pool_size = None
        target_size = None
        folder = dataname+'/TU_PU_r0_'+str(int(r_0*100))+'_s'+str(ne_params['s2'])+'_split'+str(nr_split)+'_ne'+str(nr_ne)+'_stp'+str(stp_s)+'_bgt'+str(bgt_k)
    elif Case == 2:
        bgt_k = stp_s*5  # changing parameter
        pool_size = stp_s*10  # changing parameter
        target_size = None
        folder = dataname+'/TU_r0_'+str(int(r_0*100))+'_s'+str(ne_params['s2'])+'_split'+str(nr_split)+'_ne'+str(nr_ne)+'_stp'+str(stp_s)+'_bgt'+str(bgt_k)+'_P'+str(pool_size)
    elif Case == 3:
        bgt_k = stp_s*5  # changing parameter
        pool_size = stp_s*10  # changing parameter
        target_size = stp_s*5  # changing parameter
        folder = dataname+'/r0_'+str(int(r_0*100))+'_s'+str(ne_params['s2'])+'_split'+str(nr_split)+'_ne'+str(nr_ne)+'_stp'+str(stp_s)+'_bgt'+str(bgt_k)+'_P'+str(pool_size)+'_T'+str(target_size)
    else:
        print('No such Case.')
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(folder)

    # run experiment
    res = []
    p = Pool(processes=len(strategy))
    for split_id in range(nr_split):
        folder_split = folder+'/split_'+str(split_id)
        if not os.path.exists(folder_split):
            os.makedirs(folder_split)
        res_split_id = one_split_all_s(p, folder_split, full_A,
                                    r_0, stp_s, Case, bgt_k,
                                    pool_size, target_size,
                                    strategy, split_id, nr_ne)
        res.append(res_split_id)
    print(res)


    # print averaged the scores
    avg_scores = {s: [] for s in labels}
    print(avg_scores)
    for split_id in range(nr_split):
        for s_id in range(len(strategy)):
            for ne_id in range(nr_ne):
                path_id = folder+'/split_'+str(split_id)+'/'+strategy[s_id]+'_ne_'+str(ne_id)+'.pkl'
                tmp = from_cache(path_id)
                if s_id <= 2:
                    avg_scores[labels[0]].append(tmp)
                else:
                    avg_scores[labels[s_id-2]].append(tmp)
    res_avg_scores = {}
    for j in range(len(labels)):
        score_j = list(np.mean(np.array(avg_scores[labels[j]]), axis=0))
        res_avg_scores[labels[j]] = score_j
    print(res_avg_scores)

    # plot the results
    l_s = len(res_avg_scores[labels[0]])
    print(l_s)
    x = np.linspace(0, l_s, l_s)
    for j in range(len(labels)):
        score_j = res_avg_scores[labels[j]]
        print(labels[j], score_j)
        plt.plot(x, score_j, label=labels[j])
    plt.legend()
    plt.savefig(folder+'/results.png')
    plt.close()
