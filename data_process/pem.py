import json
import os
import pickle
import time

import numpy as np
from sklearn.svm import SVR

from data_process.cal_scores import CalScore
from data_process.extract_phrase import sentence_bpng, ngrams_f1, sentence_ngrams


class PEM(object):
    def __init__(self, z2e_prob_dist, phrase_log_prob, scorer):
        self.z2e_prob_dist = z2e_prob_dist
        self.phrase_log_prob = phrase_log_prob
        self.scorer = scorer

    def pair_to_features(self, sentences, paraphrases):
        bpngs = [(sentence_bpng(sentence, self.phrase_log_prob, self.z2e_prob_dist),
                 sentence_bpng(paraphrase, self.phrase_log_prob, self.z2e_prob_dist)) for sentence,paraphrase in zip(sentences,paraphrases)]
        adequacy = [ngrams_f1(bpng[0], bpng[1]) for bpng in bpngs ]

        zh_ngrams = [(sentence_ngrams(sentence), sentence_ngrams(paraphrase) ) for sentence,paraphrase in zip(sentences,paraphrases)]
        dissimilarity = [ ngrams_f1(zh_ngram[0], zh_ngram[1]) for zh_ngram in zh_ngrams]

        ppl = self.scorer.get_ppl_from_lm(paraphrases)

        fluency = -np.log(ppl)

        return list(zip(adequacy, fluency, dissimilarity))

    def pem_score(self, svm, sentences, paraphrases):
        features = self.pair_to_features(sentences, paraphrases)
        scores = svm.predict(features)

        return scores

def get_svm(svm_path,data_dir):
    if not os.path.isfile(svm_path):



        with open(os.path.join(data_dir, 'train_rewrite_pairs2.tsv'), 'r',
                  encoding='utf8') as f:
            paraphrase_scores = f.read().splitlines()

        paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]

        sentences, paraphrases, scores = list(zip(*paraphrase_scores))

        features = pem.pair_to_features(sentences, paraphrases)

        # svm_path = os.path.join(data_dir, 'svm.pickle')




        # with open(os.path.join(data_dir, 'rewrite_pairs_scores.tsv'), 'r',
        #           encoding='utf8') as f:
        #     paraphrase_scores = f.read().splitlines()
        #
        # paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]
        #
        # sentences, paraphrases, scores = list(zip(*paraphrase_scores))

        svm = SVR()
        svm.fit(features, scores)

        with open(svm_path, 'wb') as f:
            pickle.dump(svm, f)

        train_scores = svm.predict(features)

        mse = np.sum([(x - y) ** 2 for x, y in zip(train_scores, scores)])/len(scores)

        train_scores = [str(s) for s in train_scores]

        path = os.path.join(data_dir, 'train_scores_{}'.format(time.strftime("%y-%m-%d_%H:%M:%S")))
        with open(path, 'w', encoding='utf8') as f:
            f.write('\n'.join(train_scores))

        print(train_scores)
        print('mse:{}'.format(mse))

    else:
        with open(svm_path, 'rb') as f:
            svm = pickle.load(f)

    return svm

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    scorer = CalScore('/data/share/liuchang/car_comment/mask/mask_comments/data_process/unigram_probs_model.json')

    data_dir = '/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask'

    with open('/data/share/liuchang/car_comment/mask/z2e_probdist_pruned_1e6_0.01.json', 'r') as f:
        z2e_prob_dist = json.load(f)
    print('z2e_prob_dist loaded.')

    with open('/data/share/liuchang/car_comment/mask/phrase_prob_pruned_1e6.json', 'r') as f:
        phrase_log_prob = json.load(f)
    print('phrase_log_prob loaded.')

    pem = PEM(z2e_prob_dist, phrase_log_prob, scorer)

    with open(os.path.join(data_dir, 'raw_test_corpus_only_mask'), 'r',
              encoding='utf8') as f:
        raw = f.read().splitlines()

    with open(os.path.join(data_dir, 'modify_unchanged_context/result'), 'r',
              encoding='utf8') as f:
        rewrites = f.read().splitlines()

    features=pem.pair_to_features(raw,rewrites)

    features=[[str(s) for s in l] for l in features]

    features=['\t'.join(s) for s in features]

    with open(os.path.join(data_dir, 'features_{}'.format(time.strftime("%y-%m-%d_%H:%M:%S"))), 'w',
              encoding='utf8') as f:
        f.write('\n'.join(features))

    # svm_path = '/data/share/liuchang/car_comment/mask/svm.pickle'
    # svm=get_svm(svm_path,data_dir)
    #
    # with open(os.path.join(data_dir, 'test_rewrite_pairs.tsv'), 'r',
    #           encoding='utf8') as f:
    #     paraphrase_scores = f.read().splitlines()
    #
    # paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]
    #
    # sentences, paraphrases, scores = list(zip(*paraphrase_scores))
    #
    # test_scores = pem.pem_score(svm, sentences, paraphrases)
    #
    # mse = np.sum([(x - y) ** 2 for x, y in zip(test_scores, scores)])/len(scores)
    #
    # test_scores = [str(s) for s in test_scores]
    #
    # path = os.path.join(data_dir, 'test_scores_{}'.format(time.strftime("%y-%m-%d_%H:%M:%S")))
    # with open(path, 'w', encoding='utf8') as f:
    #     f.write('\n'.join(test_scores))
    #
    # print(test_scores)
    # print('mse:{}'.format(mse))

