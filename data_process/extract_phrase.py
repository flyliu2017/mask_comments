# coding=utf8

import pickle
import time

from nltk.probability import FreqDist
from nltk.util import ngrams
import numpy as np
import os
from collections import defaultdict

from data_process.cal_scores import CalScore
from sklearn.svm import SVR


def aligns_to_dicts(aligns):
    aligns = [s.strip().split(' ') for s in aligns]
    aligns = [[s.split('-') for s in l] for l in aligns]

    en_zh_dict = []
    zh_en_dict = []

    for l in aligns:
        z2e = dict()
        e2z = dict()
        if l != [[""]]:
            for n in l:
                if n[0]:
                    zh = int(n[0])
                    en = int(n[1])
                    z2e[zh] = z2e.get(zh, set())
                    z2e[zh].add(en)
                    e2z[en] = e2z.get(en, set())
                    e2z[en].add(zh)

        en_zh_dict.append(e2z)
        zh_en_dict.append(z2e)

    return zh_en_dict, en_zh_dict


def get_values_set(d, keys):
    s = set()
    for k in keys:
        s.update(d.get(k, set()))
    return s


def phrase_pairs_from_dicts(s2t_dict, t2s_dict, s_txt, t_txt):
    pairs = []
    for t2s, s2t, s, t in zip(t2s_dict, s2t_dict, s_txt, t_txt):
        l = []
        for length in range(1, 5):
            for i in range(len(s) - length):
                s_index = set(range(i, i + length))
                t_set = get_values_set(s2t, s_index)

                if not t_set:
                    continue

                t_min = min(t_set)
                t_max = max(t_set)
                t_index = set(range(t_min, t_max + 1))
                s_reverse = get_values_set(t2s, t_index)

                if not s_reverse or s_reverse - s_index:
                    continue

                l.append((' '.join(s[i:i + length]), ' '.join(t[t_min:t_max + 1])))

        pairs.append(l)

    return pairs


def pairs_to_phrase_freq(pairs, out_dir,min_phrase_prob=0,min_z2e_prob=0):
    # 太消耗内存
    # phrases = [n[0] for l in pairs for n in l]
    # phrase_log_prob = FreqDist(phrases)
    #
    # z2e_fd = FreqDist(n for l in pairs for n in l)
    # z2e_prob_dist = {}
    # for zh, en in z2e_fd:
    #     z2e_prob_dist[zh] = z2e_prob_dist.get(zh, [])
    #     z2e_prob_dist[zh].append([en, z2e_fd[(zh, en)]])
    # for zh in z2e_prob_dist:
    #     for l in z2e_prob_dist[zh]:
    #         l[1] = l[1] / phrase_log_prob[zh]
    #
    # n = phrase_log_prob.N() + 0.5
    # for k in phrase_log_prob:
    #     phrase_log_prob[k] = np.log(phrase_log_prob[k] / n)
    # phrase_log_prob['<oov>'] = np.log(0.5 / n)


    z2e_prob_dist = {}
    phrase_log_prob = FreqDist()
    for pair_list in pairs:
        for zh, en in pair_list:
            print(time.strftime("%H:%M:%S"))
            fd = z2e_prob_dist.setdefault(zh, FreqDist())
            fd.update([en])

    N=sum([z2e_prob_dist[zh].N() for zh in z2e_prob_dist])
    min_phrase_num=np.ceil(N*min_phrase_prob)

    for zh in z2e_prob_dist:
        print(time.strftime("%H:%M:%S"))
        n = z2e_prob_dist[zh].N()
        if n>=min_phrase_num:
            phrase_log_prob[zh] = n
        for en in z2e_prob_dist[zh]:
            prob=z2e_prob_dist[zh][en]/n
            if prob>=min_z2e_prob:
                z2e_prob_dist[zh][en] =prob

    n = phrase_log_prob.N()+0.5
    for k in phrase_log_prob:
        phrase_log_prob[k] = np.log(phrase_log_prob[k] / n)
    phrase_log_prob['<oov>']=np.log(0.5/n)

    # with open(os.path.join(out_dir,'phrase_frequency.json'), 'w', encoding='utf8') as f:
    #     json.dump(phrase_log_prob,f,indent=4,ensure_ascii=False)
    # with open(os.path.join(out_dir,'zh2en_prob.json'), 'w', encoding='utf8') as f:
    #     json.dump(z2e_prob_dist,f,indent=4,ensure_ascii=False)

    path=os.path.join(out_dir, 'phrase_log_prob.pickle')
    if os.path.isfile(path):
        path=path+time.strftime("%y-%m-%d_%H:%M:%S")
    with open(path, 'wb') as f:
        pickle.dump(phrase_log_prob, f)

    path=os.path.join(out_dir, 'zh2en_prob.pickle')
    if os.path.isfile(path):
        path=path+time.strftime("%y-%m-%d_%H:%M:%S")
    with open(path, 'wb') as f:
        pickle.dump(phrase_log_prob, f)

    return phrase_log_prob, z2e_prob_dist


def sentence_segmentation(sentence, phrase_log_prob):
    sentence = sentence.strip()
    if not sentence:
        raise ValueError('Empty sentence!')

    words = sentence.split()
    words=list(filter(None,words))
    length = len(words)

    prob_list = [[] for _ in range(length)]
    prob_list[0] = [[0], phrase_log_prob[words[0]]]
    for i in range(1, length):
        max_prob = prob_list[i - 1][1] + phrase_log_prob[words[i]]
        prob_list[i] = [prob_list[i - 1][0] + [i], max_prob]
        for j in range(max(0, i - 4), i - 1):
            if ' '.join(words[j + 1:i + 1]) in phrase_log_prob:
                prob = phrase_log_prob[' '.join(words[j + 1:i + 1])] + prob_list[j][1]
                if prob > max_prob:
                    prob_list[i] = [prob_list[j][0] + [j+1], prob]
                    max_prob = prob

    segment_index = prob_list[-1][0] + [length]
    phrases = [' '.join(words[segment_index[i]:segment_index[i + 1]]) for i in range(len(segment_index) - 1)]

    return phrases, np.exp(prob_list[-1][1])


# def pairs_to_phrase_translation_freq(pairs):
#     z2e_fd = ConditionalFreqDist(n for l in pairs for n in l)
#     z2e_prob_dist = ConditionalProbDist(z2e_fd, MLEProbDist)
#     return z2e_prob_dist

def bag_of_pivot_language_ngrams(pivot_language_phrases_freqdist):
    ngrams_bag = [FreqDist() for _ in range(4)]
    ngrams_at_head = [FreqDist() for _ in range(4)]

    if pivot_language_phrases_freqdist:
        ngrams_tail, ngrams_at_head_tail = bag_of_pivot_language_ngrams(pivot_language_phrases_freqdist[1:])
        first_phrase_freqdist = pivot_language_phrases_freqdist[0]
        for phrase in first_phrase_freqdist:
            prob = first_phrase_freqdist[phrase]
            phrase_length = len(phrase)

            for length in range(1, 5):
                if length <= phrase_length:
                    ngrams_at_head[length - 1].update({phrase[:length + 1]: prob})
                else:
                    tail = ngrams_at_head_tail[length - 1 - phrase_length]
                    l = {phrase + p: prob * tail[p] for p in tail}
                    ngrams_at_head[length - 1].update(l)

            for i in range(1, phrase_length):
                for length in range(1, 5):
                    if length + i <= phrase_length:
                        ngrams_bag[length - 1].update({phrase[i:length + i]: prob})
                    else:
                        tail = ngrams_at_head_tail[length - 1 - phrase_length + i]
                        l = {phrase[i:] + p: prob * tail[p] for p in tail}
                        ngrams_bag[length - 1].update(l)

        for i in range(4):
            ngrams_bag[i].update(ngrams_at_head[i])
            ngrams_bag[i].update(ngrams_tail[i])

    return ngrams_bag, ngrams_at_head


def sentence_bpng(sentence, phrase_log_prob, z2e_prob_dist):
    print(str('pid:{} ||'.format(os.getpid())),time.strftime("%y-%m-%d_%H:%M:%S"))

    phrases, _ = sentence_segmentation(sentence, phrase_log_prob)
    ngrams_bag, _ = bag_of_pivot_language_ngrams([z2e_prob_dist[phrase] if phrase in z2e_prob_dist else {phrase:1.0} for phrase in phrases])
    bpng = {}
    for i in range(4):
        bpng.update(ngrams_bag[i])
    return bpng

def sentence_ngrams(sentence):
    print(str('pid:{} ||'.format(os.getpid())),time.strftime("%y-%m-%d_%H:%M:%S"))
    sentence = sentence.strip()
    if not sentence:
        raise ValueError('Empty sentence!')
    
    words=sentence.split(' ')
    ngrams_bag=FreqDist()
    for i in range(4):
        ngrams_bag.update(ngrams(words,i+1))

    return ngrams_bag

def ngrams_f1(origin_ngrams: dict, paraphrase_ngrams: dict):
    origin_total = sum(origin_ngrams.values())
    paraphrase_total = sum(paraphrase_ngrams.values())

    origin_set = set(origin_ngrams.keys())
    paraphrase_set = set(paraphrase_ngrams.keys())

    inter = origin_set & paraphrase_set
    inter_weights = sum([min(origin_ngrams[w], paraphrase_ngrams[w]) for w in inter])

    precision = inter_weights / origin_total
    recall = inter_weights / paraphrase_total

    if not ( precision and recall):
        f1=0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1

def random_sentence_from_unigram_model(unigram:dict,max_length):
    words=list(unigram.keys())
    probs=list(unigram.values())
    n=sum(probs)
    probs=[p/n for p in probs]
    return ' '.join(np.random.choice(words,max_length,p=probs))


if __name__ == '__main__':

    # if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    data_dir='/nfs/users/liuchang/berkeleyaligner/output_chinese'
    pairs_pickle=os.path.join(data_dir,'training.en-ch.pairs.pickle')

    if not os.path.isfile(pairs_pickle):
        with open('/nfs/users/liuchang/berkeleyaligner/output_chinese/itg_input/supervised-train/train.align', 'r', encoding='utf8') as f:
            aligns = f.readlines()

        zh_en_dict, en_zh_dict = aligns_to_dicts(aligns)

        with open('/nfs/users/liuchang/berkeleyaligner/output_chinese/itg_input/supervised-train/train.ch', 'r', encoding='utf8') as f:
            zhtxt = f.read().splitlines()
        with open('/nfs/users/liuchang/berkeleyaligner/output_chinese/itg_input/supervised-train/train.en', 'r', encoding='utf8') as f:
            entxt = f.read().splitlines()

        zhtxt = [s.split(' ') for s in zhtxt]
        entxt = [s.split(' ') for s in entxt]

        pairs = phrase_pairs_from_dicts(zh_en_dict, en_zh_dict, zhtxt, entxt)

        with open(pairs_pickle, 'wb') as f:
            pickle.dump(pairs,f)

    else:
        with open(pairs_pickle, 'rb') as f:
            pairs=pickle.load(f)

    fd, z2e_prob_dist = pairs_to_phrase_freq(pairs, out_dir=data_dir)

    # keys = list(z2e_prob_dist.keys())
    #
    # sentence='颜值 ， 动力 ， 颜值 ， 动力 ， 颜值 ， 动力 ， 重要 的 事情 说 三遍'
    # paraphrase=' 颜值 ， 动力 ， 颜值 ， 动力 ， 颜值 ，  动力  ， 重要 的 事情 说 三遍 ～'
    #
    # origin_bpng=sentence_bpng(sentence,fd,z2e_prob_dist)
    # paraphrase_bpng=sentence_bpng(paraphrase,fd,z2e_prob_dist)
    #
    # adequacy=ngrams_f1(origin_bpng,paraphrase_bpng)
    #
    # ngrams_bag, ngrams_at_head = bag_of_pivot_language_ngrams(
    #     [z2e_prob_dist[keys[80]], z2e_prob_dist[keys[79]], z2e_prob_dist[keys[52]], z2e_prob_dist[keys[57]]])
    #
    # print(ngrams_bag)
    # print(ngrams_at_head)

    scorer = CalScore('/nfs/users/liuchang/car_comment/mask/mask_comments/data_process/unigram_probs_model.json')


    with open('/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask/rewrite_pairs_scores.tsv', 'r',
              encoding='utf8') as f:
        paraphrase_scores = f.read().splitlines()

    paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]

    sentences,paraphrases,scores=list(zip(*paraphrase_scores))

    # svm_path = os.path.join(data_dir, 'svm.pickle')
    # svm_path = '/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask/svm.pickle'
    #
    # if not os.path.isfile(svm_path):
    #     svm = SVR()
    #     svm.fit(features,scores)
    #
    #     with open(svm_path, 'wb') as f:
    #         pickle.dump(svm,f)
    # else:
    #     with open(svm_path, 'rb') as f:
    #         svm=pickle.load(f)
    #
    # pem_scores=pem.pem_score(svm,sentences,paraphrases)
    #
    # print(pem_scores)

