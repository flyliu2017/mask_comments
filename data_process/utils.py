import re
import sacrebleu
import numpy as np

from data_process.cal_scores import CalScore


def length_selection(string, min_words=5, max_words=40):
    l = string.split(' ')
    l = [s for s in l if s.strip() != '']
    l = l if min_words <= len(l) <= max_words else []
    return ' '.join(l)

def phrase_selction(string, min_phrase=5, max_phrase=10):
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

def read_to_dict(path,sep,value_type):
    with open(path, 'r') as f:
        l = f.read().splitlines()
    l=[s.split(sep) for s in l]
    d={}
    for key,value in l:
        d[key]=value_type(value)
    return d


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
    slor=[scorer.cal_slor_with_entropy(r.split('__')[-1],float(e)) for r,e in zip(results,entropy)]
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

def extract_keywords(vectorizer, feature_names, string, keywords_file=None, ratio=0.3):
    if not string:
        return ''

    words=string.split(' ')
    words=[w for w in words if w.strip() != '']

    num=int(round(len(words)*ratio))
    num=max(1,num)

    keywords=[]

    if keywords_file:
        keywords_dict=read_to_dict(keywords_file,'\t',float)
        keywords=[(word,keywords_dict[word]) for word in words if word in keywords_dict]

        if len(keywords)>num:
            keywords.sort(key= lambda n:n[1],reverse=True)
            keywords=keywords[:num]

    keywords = [n[0] for n in keywords]

    if len(keywords)<num:
        tfidf = vectorizer.transform([string])

        z = list(zip(tfidf.data, tfidf.indices))
        z.sort(key=lambda n: n[0], reverse=True)
        indexes = [n[1] for n in z[:num]]
        keywords.extend(feature_names[indexes])


    return ' '.join(keywords) + ' [sep] '