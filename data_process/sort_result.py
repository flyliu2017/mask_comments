from data_process.cal_scores import CalScore
from data_process.utils import read_to_list
import os
from gensim.models import Word2Vec

def get_slor(scorer : CalScore, results, corpus):
    no_prefix=[r.split('__.+?__')[-1] for r in results]
    ppls=[ scorer.get_ppl_from_lm([s])[0] for s in no_prefix]
    slor=[scorer.cal_slor_with_ppl(r,p) for r,p in zip(no_prefix,ppls)]
    
    return slor

def get_wmd(corpus_path, features, predictions):
    model=Word2Vec(corpus_file=corpus_path)
    wmd=[model.wv.wmdistance(f,p) for f,p in zip(features,predictions)]
    
    return wmd

if __name__ == '__main__':

    data_dir='/data/share/liuchang/car_comment/mask/p5_p10/keywords/whole_sentence'
    results=os.path.join(data_dir,'result')
    results=read_to_list(results)
    # entropy_and_result=read(os.path.join(data_dir, 'p5_p10/entropy_and_result'))
    # entropy_and_result=[s.split('|||') for s in entropy_and_result]
    # entropy=[n[0] for n in entropy_and_result]
    # results=[n[1] for n in entropy_and_result]

    features=read_to_list(os.path.join(data_dir, 'raw_test_corpus_p5_p10'))
    
    corpus_path = '/data/share/liuchang/car_comment/mask/selected_str_5_80.txt'
    corpus=read_to_list(corpus_path)

    sort='wmd'
    reverse=True
    
    if sort=='slor':
        scorer=CalScore('unigram_model.json')
        num=3000
        score=get_slor(scorer, results[:num], features[:num])

    elif sort=='wmd':
        reverse=False
        score=get_wmd(corpus_path, features, results)
    
    else:
        raise ValueError('Wrong sort type.')

    results_with_score = list(zip(results, features, score))
    results_with_score.sort(key=lambda n: n[-1], reverse=reverse)
    sorted_result = [r + '\t' + str(s) for r, _, s in results_with_score]
    sorted_features = [f for _, f, _ in results_with_score]
    with open(os.path.join(data_dir, 'results_sorted_by_{}'.format(sort)), 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted_result))
    with open(os.path.join(data_dir, 'comparation_sorted_by_{}'.format(sort)), 'w', encoding='utf8') as f:
        for r, c in zip(sorted_result, sorted_features):
            f.write(r + '\n' + c + '\n\n')
        


