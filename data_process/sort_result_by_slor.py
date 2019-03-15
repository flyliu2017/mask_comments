from data_process.cal_scores import CalScore
from data_process.processor import Processor
import os

def sort_by_slor(scorer : CalScore,results,cross_entropy):
    slor=[scorer.cal_slor_with_entropy(r.split('__')[-1],c) for r ,c in zip(results, cross_entropy)]
    results_with_score=[r+'\t'+str(s) for r,s in zip(results,slor)]
    return results_with_score

def sort_by_slor2(scorer : CalScore, results,entropy, corpus):
    slor=[scorer.cal_slor_with_entropy(r.split('__')[-1],float(e)) for r,e in zip(results,entropy)]
    results_with_slor=list(zip(results,corpus,slor))
    results_with_slor.sort(key=lambda n:n[-1],reverse=True)

    return results_with_slor


data_dir='/data/share/liuchang/car_comment/mask/'
# results=os.path.join(data_dir,'result')
entropy_and_result=Processor.read(os.path.join(data_dir, 'p5_p10/entropy_and_result'))
entropy_and_result=[s.split('|||') for s in entropy_and_result]
entropy=[n[0] for n in entropy_and_result]
results=[n[1] for n in entropy_and_result]

corpus=Processor.read(os.path.join(data_dir,'p5_p10/raw_test_corpus_p5_p10'))

scorer=CalScore('unigram_model.json')
results_with_slor=sort_by_slor2(scorer, results,entropy,corpus)
sorted_result=[r + '\t' + str(s) for r,_, s in results_with_slor]
sorted_corpus=[ c for _,c, _ in results_with_slor]
with open(os.path.join(data_dir,'sorted_results'), 'w', encoding='utf8') as f:
    f.write('\n'.join(sorted_result))
with open(os.path.join(data_dir,'sorted_compare'), 'w', encoding='utf8') as f:
    for r,c in zip(sorted_result,sorted_corpus):
        f.write(r+'\n'+c+'\n\n')

