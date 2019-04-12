import os
import pickle
import time

import numpy as np
from sklearn.svm import SVR

from data_process.cal_scores import CalScore
from data_process.extract_phrase import sentence_bpng, ngrams_f1, sentence_ngrams


class PEM(object):
    def __init__(self,z2e_prob_dist,phrase_log_prob,scorer):
        self.z2e_prob_dist=z2e_prob_dist
        self.phrase_log_prob=phrase_log_prob
        self.scorer=scorer

    def pair_to_features(self,sentence,paraphrase):

        bpngs=[sentence_bpng(sentence,self.phrase_log_prob,self.z2e_prob_dist),sentence_bpng(paraphrase,self.phrase_log_prob,self.z2e_prob_dist)]
        adequacy=ngrams_f1(bpngs[0],bpngs[1])

        zh_ngrams=[sentence_ngrams(sentence),sentence_ngrams(paraphrase)]
        dissimilarity=ngrams_f1(zh_ngrams[0],zh_ngrams[1])

        ppl=self.scorer.get_ppl_from_lm([paraphrase])[0]

        fluency=-np.log(ppl)

        return adequacy,fluency,dissimilarity

    def pem_score(self,svm,sentences,paraphrases):
        features=[self.pair_to_features(s,p) for s ,p in zip(sentences,paraphrases)]
        scores=svm.predict(features)

        return scores

if __name__ == '__main__':
    scorer = CalScore('/data/share/liuchang/car_comment/mask/mask_comments/data_process/unigram_probs_model.json')

    data_dir='/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask'
    with open(os.path.join(data_dir,'rewrite_pairs_scores.tsv'), 'r',
              encoding='utf8') as f:
        paraphrase_scores = f.read().splitlines()

    paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]

    sentences, paraphrases, scores = list(zip(*paraphrase_scores))


    with open('/data/share/liuchang/car_comment/mask/zh2en_prob.pickle', 'rb') as f:
        z2e_prob_dist = pickle.load(f)

    with open('/data/share/liuchang/car_comment/mask/phrase_log_prob.pickle', 'rb') as f:
        phrase_log_prob = pickle.load(f)

    pem = PEM(z2e_prob_dist, phrase_log_prob, scorer)

    features = [pem.pair_to_features(s, p) for s, p in zip(sentences, paraphrases)]

    # svm_path = os.path.join(data_dir, 'svm.pickle')
    svm_path = '/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask/svm.pickle'

    if not os.path.isfile(svm_path):
        svm = SVR()
        svm.fit(features, scores)

        with open(svm_path, 'wb') as f:
            pickle.dump(svm, f)
    else:
        with open(svm_path, 'rb') as f:
            svm = pickle.load(f)

    pem_scores = pem.pem_score(svm, sentences, paraphrases)

    path=os.path.join(data_dir,'pem_scores_{}'.format(time.strftime("%y-%m-%d_%H:%M:%S")))

    print(pem_scores)