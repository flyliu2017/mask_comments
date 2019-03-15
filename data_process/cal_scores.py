import numpy as np
import json
from rouge import Rouge
from nltk.probability import FreqDist

class CalScore(object):
    def __init__(self, model_path='ngram_probs_model.json'):
        self.rouge = Rouge()  # codes from https://github.com/pltrdy/rouge
        self.unigram_probs = json.load(open(model_path))

    def cal_slor_with_ppl(self, rewrite_tokens, ppl):
        len_tokens = len(rewrite_tokens.split())
        ## 计算SLOR分数: (-ln(ppl)-ln(P(S)))/|S|
        unigram_probs = 1.0
        for token in rewrite_tokens.split():
            token = token.lower()
            if token in self.unigram_probs:
                token_prob = self.unigram_probs[token]
            else:
                token_prob = self.unigram_probs['<unk>']
                print('assert token: {} not found...'.format(token))

            unigram_probs *= token_prob
        slor_score = (-np.log(ppl) - np.log(unigram_probs)) / len_tokens
        return slor_score

    def cal_slor_with_entropy(self, rewrite_tokens, entropy_loss):
        len_tokens = len(rewrite_tokens.split())
        ## 计算SLOR分数: (-entropy_los - ln(P(S)))/|S|
        unigram_probs = 1.0
        for token in rewrite_tokens.split():
            token = token.lower()
            if token in self.unigram_probs:
                token_prob = self.unigram_probs[token]
            else:
                token_prob = self.unigram_probs['<unk>']
                print('assert token: {} not found...'.format(token))

            unigram_probs *= token_prob
        slor_score = (-entropy_loss - np.log(unigram_probs)) / len_tokens
        return slor_score

    def cal_rouge(self, rewrite_tokens, original_tokens):
        rouge_scores = self.rouge.get_scores(rewrite_tokens, original_tokens, avg=True)  # rouge scores
        rouge_score = rouge_scores['rouge-2']['f']
        return rouge_score

    def generate_unigram_model(self,corpus,vocab):
        fd=FreqDist(corpus)
        n=fd.N()
        d=dict(fd)
        d['<unk>']=0
        for k in d.keys():
            if not k in vocab:
                d['<unk>']+=d[k]
                d.pop(k)
        for k in d.keys():
            d[k]=d[k]/n
        with open('unigram_model.json', 'w', encoding='utf8') as f:
            json.dump(d,f)

def demo():
    unigram_probs_filepath = './ngram_probs_model.json'
    model = CalScore(unigram_probs_filepath)

    original_tokens = "接缝 控制 得 很 好 ， 至于 外观 是 不 惊艳 但是 比较 耐 看 的 那种"
    rewrite_tokens = "一般 开 的 比较 少 觉得 动力 还是 有 点 弱"

    ## rouge score
    rouge_score = model.cal_rouge(rewrite_tokens, original_tokens)
    print('ROUGE-2 score: {}'.format(rouge_score))

    entropy_loss = 0.1  ## entropy loss of rewrite sequence
    ppl = np.exp(entropy_loss)

    ## SLOR score
    slor_score = model.cal_slor_with_entropy(rewrite_tokens, entropy_loss)
    slor_score_ppl = model.cal_slor_with_ppl(rewrite_tokens, ppl)
    assert(slor_score == slor_score_ppl)
    print('SLOR score: {}'.format(slor_score))