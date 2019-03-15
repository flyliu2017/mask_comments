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


def read(path):
    with open(path, 'r') as f:
        l = f.read().splitlines()
    return l


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