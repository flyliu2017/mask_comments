from data_process.cal_scores import CalScore
from data_process.utils import read_to_list
import os
import argparse
from gensim.models import Word2Vec
import re
import numpy as np

def get_slor(scorer : CalScore, results):
    batch=10
    times=np.ceil(len(results)/batch)
    times=int(times)
    no_prefix=[re.split('__.+?__',r)[-1].strip() for r in results]
    ppls=[]
    for i in range(times):
        ppls.extend(scorer.get_ppl_from_lm(no_prefix[i*batch:(i+1)*batch]))

    slor=[scorer.cal_slor_with_ppl(r,p) for r,p in zip(no_prefix,ppls)]
    
    return slor

def get_wmd(corpus_path, features, predictions):
    model=Word2Vec(corpus_file=corpus_path)
    wmd=[model.wv.wmdistance(f, p) for f, p in zip(features, predictions)]
    
    return wmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sort",
                        choices=["slor", "wmd"],
                        help="Run type.")
    parser.add_argument("--data_dir",
                        default='/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask',
                        help="The data directory.")
    parser.add_argument("--result_file",
                        default='modify/result',
                        help="The result file.")
    parser.add_argument("--features_file",
                        default='no_mask_test_corpus_only_mask',
                        help="The features fle.")
    parser.add_argument("--suffix",
                        default='only_mask',
                        help="The suffix of filename.")
    parser.add_argument("--reverse", default=False,
                        dest='reverse',
                        action='store_true',
                        help="True for descending.")
    parser.add_argument("--num", default=3000, type=int,
                        help="Number of results to be sorted. 0 for whole results.")
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    results = os.path.join(data_dir, args.result_file)
    results = read_to_list(results)
    features = read_to_list(os.path.join(data_dir, args.features_file))
    corpus_path = '/data/share/liuchang/car_comment/mask/selected_str_5_80.txt'

    sort = args.sort
    reverse = args.reverse

    if args.num:
        num = args.num
    else:
        num = len(results)

    if sort == 'slor':
        scorer = CalScore('/data/share/liuchang/car_comment/mask/mask_comments/data_process/unigram_model.json')
        score = get_slor(scorer, results[:num])

    else:
        score = get_wmd(corpus_path, features[:num], results[:num])

    # data_dir = os.path.join(data_dir, 'wmd_whole')

    results_with_score = list(zip(results, features, score))
    results_with_score.sort(key=lambda n: n[-1], reverse=reverse)
    sorted_result = [r + '\t' + str(s) for r, _, s in results_with_score]
    sorted_features = [f for _, f, _ in results_with_score]
    with open(os.path.join(data_dir, 'results_sorted_by_{}'.format(sort)), 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted_result))
    with open(os.path.join(data_dir, 'comparation_sorted_by_{}'.format(sort)), 'w', encoding='utf8') as f:
        for r, c in zip(sorted_result, sorted_features):
            f.write(r + '\n' + c + '\n\n')
    scores = [n[-1] for n in results_with_score]
    average = sum(scores) / len(scores)
    scores.append(average)
    print("average score: {}".format(average))
    scores = [str(s) for s in scores]
    with open(os.path.join(data_dir, sort), 'w', encoding='utf8') as f:
        f.write('\n'.join(scores))


if __name__ == '__main__':

    main()
    

