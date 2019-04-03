import re
import sacrebleu
import numpy as np
import time

from data_process.cal_scores import CalScore




def length_selection(string, min_words=5, max_words=40):
    l = string.split(' ')
    l = [s for s in l if s.strip() != '']
    l = l if min_words <= len(l) <= max_words else []
    return ' '.join(l)

def phrase_selection(string, min_phrase=5, max_phrase=10):
    l = re.split('[，。]', re.sub('[!?;！？；]|… …|…', '，', string))
    l = [s for s in l if s.strip() != '']
    if len(l) < min_phrase or len(l) > max_phrase:
        return ''
    else:
        return '，'.join(l)
    

def compare_mask(preds, labels, corpus, output='compare'):
    if len(preds) != len(labels) or len(labels) != len(corpus):
        raise ValueError("predictions,labels and corpus should have same length.")

    corpus=[s.split('[sep]')[-1] for s in corpus]
    result = []
    for p, l, c in zip(preds, labels, corpus):
        result.append(c.replace('mask', l + " | " + p))
    with open(output, 'w', encoding='utf8') as f:
        f.write('\n'.join(result))


def compare_result(preds, labels, corpus, output='compare'):
    if len(preds) != len(labels) or len(labels) != len(corpus):
        raise ValueError("predictions,labels and corpus should have same length.")

    corpus=[s.replace('<mask>',l) for s,l in zip(corpus,labels)]
    corpus=[s.split('[sep]')[-1] for s in corpus]
    corpus=[ s+'\n' if not s.endswith('\n') else s for s in corpus]
    preds=[ s+'\n' if not s.endswith('\n') else s for s in preds]
    with open(output, 'w', encoding='utf8') as f:
        for r, c in zip(preds, corpus):
            f.writelines([r, c, '\n'])


def read_to_list(path):
    with open(path, 'r') as f:
        l = f.read().splitlines()
    return l

def read_to_dict(path,sep,value_type,range):
    with open(path, 'r') as f:
        l = f.read().splitlines()
    if range:
        l=l[:range]
    l=[s.split(sep) for s in l]
    d={}
    for key,value in l:
        d[key]=value_type(value)
    return d

STOP_WORDS=read_to_list('/data/share/liuchang/car_comment/mask/stop_words')

def cal_bleu(predictions, labels, output="bleu"):
    if len(predictions) != len(labels):
        raise ValueError('The number of Predictions and labels should be same.')
    results = []
    for p, l in zip(predictions, labels):
        results.append(sacrebleu.sentence_bleu(p, l))
    with open(output, 'w', encoding='utf8') as f:
        f.write('\n'.join([str(r) for r in results]))
        ave = sum(results) / len(predictions)
        f.write('\n' + str(ave))
    print(ave)


def file_bleu(pred_path, labels_path, output="bleu"):
    with open(pred_path, 'r', encoding='utf8') as f:
        predictions = f.readlines()
    with open(labels_path, 'r', encoding='utf8') as f:
        labels = f.readlines()
    cal_bleu(predictions, labels, output)

def slice_and_save(text_list, shuffle_index, slice_ratios, paths):
    if len(paths) != len(slice_ratios) + 1:
        print('wrong num of output paths!')
        exit(1)
    cum_sum = np.cumsum(slice_ratios)
    if cum_sum[np.logical_or(cum_sum < 0, cum_sum > 1)].size:
        print('wrong slide_ratio!')
        exit(1)
    l = len(text_list)
    slide_num = [0] + [int(l * cum_sum[i]) for i in range(len(cum_sum))] + [l]

    # 使用numpy.random.shuffle容易内存溢出，使用索引重建python列表可避免
    shuffle_list = [text_list[i] for i in shuffle_index]
    for i in range(len(slide_num) - 1):
        with open(paths[i], 'w', encoding='utf8') as f:
            f.write('\n'.join(shuffle_list[slide_num[i]:slide_num[i + 1]]))


def sort_by_slor(scorer : CalScore, results,entropy, corpus,output):
    slor=[scorer.cal_slor_with_entropy(r.split('__.+?__')[-1],float(e)) for r,e in zip(results,entropy)]
    results_with_slor=list(zip(results,corpus,slor))

    results_with_slor.sort(key=lambda n:n[-1],reverse=True)

    sorted_result = [r + '\t' + str(s) for r, _, s in results_with_slor]
    sorted_corpus = [c for _, c, _ in results_with_slor]
    with open(output, 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted_result))
    with open(output, 'w', encoding='utf8') as f:
        for r, c in zip(sorted_result, sorted_corpus):
            f.write(r + '\n' + c + '\n\n')

    return results_with_slor

def extract_keywords(vectorizer, feature_names, string, word_tfidf:dict=None,corpus_keywords:dict =None,stop_words=None,ratio=0.3):
    """

    :param vectorizer: tfidfvectorizer
    :param feature_names: words list from vectorizer.get_feature_names()
    :param string: string to be extracted
    :param word_tfidf: a dict contain all words in corpus, { word : tfidf }
    :param corpus_keywords: the keywords list that we will search keywords in it firstly.
    :param stop_words: stop words
    :param ratio: the number of keywords is round( NUM * ratio), NUM is the number of words in string.
    :return:
    """

    print(time.strftime("%H:%M:%S"))

    if not string:
        return ''

    if not corpus_keywords:
        corpus_keywords = {}

    feature_names=np.array(feature_names)

    words=string.split(' ')
    words=[w for w in words if w.strip() != '']

    words_set=set(words)
    stop_words_set={w for w in words_set if w in stop_words}
    

    num=int(round(len(words_set)*ratio))
    num=max(1,num)

    keywords=[]

    if corpus_keywords:
        keywords=[(word,corpus_keywords[word]) for word in words_set if word in corpus_keywords]

        if len(keywords)>num:
            keywords.sort(key= lambda n:n[1],reverse=True)
            keywords=keywords[:num]

    keywords = [n[0] for n in keywords]
    
    
    
    if len(keywords)<num:
        tfidf = vectorizer.transform([string])

        zipped = list(zip(tfidf.data, tfidf.indices))
        zipped=[n for n in zipped if not feature_names[n[1]] in keywords]
        zipped.sort(key=lambda n: n[0], reverse=True)

        indexes = [n[1] for n in zipped[:num-len(keywords)]]
        keywords.extend(feature_names[indexes])
    
    if len(keywords)<num:
        remain = words_set - set(keywords) - stop_words_set
        if len(remain):
            keywords.extend(topn_words_from_dict(word_tfidf,  num-len(keywords), remain))

    if len(keywords)<num:
        if len(stop_words_set):
            keywords.extend(topn_words_from_dict(word_tfidf,  num-len(keywords), stop_words_set))

    # 将关键词按它们在语句中出现的顺序排列
    keywords=[word for word in words if word in keywords]

    return ' '.join(keywords) + ' [sep] '

def topn_words_from_dict(word_score_dict, num, candidates=None, reverse=True):
    if not candidates:
        candidates=word_score_dict.keys()

    l = [(w, word_score_dict[w]) for w in candidates]
    l.sort(key=lambda n: n[1], reverse=reverse)
    l = [n[0] for n in l[:num]]
    return l



