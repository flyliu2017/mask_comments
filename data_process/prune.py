import json
import pickle
from nltk.probability import FreqDist

from nltk.probability import FreqDist
import numpy as np
import struct

# with open('/nfs/users/liuchang/car_comment/mask/zh2en_prob.pickle', 'rb') as f:
#     z2e_prob_dist = pickle.load(f)
#
# print('load z2d')
# def prune_dict(z2e_prob_dist,min_prob):
#     z2e_pruned = {}
#     for zh in z2e_prob_dist:
#         pd = z2e_prob_dist[zh]
#         fd = FreqDist()
#         z2e_pruned[zh] = fd
#         for en in pd:
#             if pd[en] >= min_prob:
#                 fd[en] = pd[en]
#         n = fd.N()
#         for en in fd:
#             fd[en] = fd[en] / n
#
#     return z2e_pruned
#
# z2e_pruned=prune_dict(z2e_prob_dist,0.01)

# with open('/nfs/users/liuchang/car_comment/mask/phrase_log_prob.pickle', 'rb') as f:
#     phrase_log_prob = pickle.load(f)

# print('load phrase_log_prob')
# def prune_phrase_prob(phrase_log_prob:FreqDist, num):
#     remain=phrase_log_prob.most_common(num)
#
#     remain=[(l,np.exp(p)) for l,p in remain]
#     n=sum([n[1] for n in remain])+0.5
#     remain={' '.join(l):np.log(p/n) for l,p in remain}
#     remain['<oov>']=np.log(0.5/n)
#     return remain

# phrase_log_prob_pruned=prune_phrase_prob(phrase_log_prob,10000000)

# print('prune done.')
# del phrase_log_prob

if __name__ == '__main__':

    with open('/nfs/users/liuchang/car_comment/mask/phrase_prob_pruned_1e7.json', 'r',encoding='utf8') as f:
        pd=json.load(f)

    with open('/nfs/users/liuchang/car_comment/mask/zh2en_prob.pickle', 'rb') as f:
        z2e_prob_dist = pickle.load(f)

    print('load z2d')

    pruned={}
    for zh in pd:
        en_prob=z2e_prob_dist[tuple(zh.split(' '))]
        pruned[zh]={' '.join(k):en_prob[k] for k in en_prob}

    print('prune done')

    with open('/nfs/users/liuchang/car_comment/mask/z2e_probdist_pruned_1e7.json', 'w',encoding='utf8') as f:
        json.dump(pruned,f)
