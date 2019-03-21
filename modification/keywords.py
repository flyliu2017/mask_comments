from data_process.utils import extract_keywords,read_to_list,read_to_dict
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
import numpy as np
from modification.change_mask import recover_mask


STOP_WORDS=read_to_list('/data/share/liuchang/car_comment/mask/stop_words')

def phrase_keywords_lists(out_dir, corpus, vectorizer, suffix):
    phrase_lists = [s.split('，') for s in corpus]
    feature_names=np.array(vectorizer.get_feature_names())

    corpus_keywords = read_to_dict('/data/share/liuchang/car_comment/mask/corpus_keywords', '\t', float, 1000)
    word_tfidf = read_to_dict('/data/share/liuchang/car_comment/mask/word_tfidf', '\t', float, None)
    keywords_lists = [[extract_keywords(vectorizer, feature_names, s,word_tfidf,corpus_keywords,stop_words=STOP_WORDS) for s in l] for l in phrase_lists]
    SEP = ' ||| '
    keywords = [SEP.join(l) for l in keywords_lists]
    with open(os.path.join(out_dir, 'phrase_keywords_lists_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(keywords))

    return keywords_lists
        
def corpus_keywords(out_dir, corpus,vectorizer):
    feature_names=np.array(vectorizer.get_feature_names())
    tfidf=vectorizer.transform(corpus).toarray()[0]
    word_with_tfidf=list(zip(feature_names,tfidf))
    word_with_tfidf.sort(key=lambda n:n[1],reverse=True)
    word_with_tfidf=[n[0]+'\t'+str(n[1]) for n in word_with_tfidf]
    
    with open(os.path.join(out_dir, 'corpus_keywords'), 'w', encoding='utf8') as f:
        f.write('\n'.join(word_with_tfidf))
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",
                        default='corpus',
                        choices=["phrase", "corpus"],
                        help="Run type.")

    parser.add_argument("--data_dir",
                        default='/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask',
                        help="The data directory.")
    parser.add_argument("--corpus_file",
                        default='/data/share/liuchang/car_comment/mask/selected_str_5_80_p5_p10.txt',
                        help="The corpus file.")
    parser.add_argument("--suffix",
                        default='',
                        help="The filename suffix.")

    parser.add_argument("--max_features", default=None, type=int,
                        help="max_features for tfidfvectorizer.")

    args = parser.parse_args()
    print(args)

    data_dir=args.data_dir
    suffix = args.suffix

    corpus = read_to_list(args.corpus_file)
    
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=STOP_WORDS, max_features=args.max_features)
    vectorizer.fit(corpus)

    run=args.run
    if 'phrase'==run:
        recover_mask(data_dir,suffix)
        
        with open(os.path.join(data_dir, 'raw_test_corpus_{}'.format(suffix)), 'r', encoding='utf8') as f:
            raw = f.read().splitlines()
            
        phrase_keywords_lists(data_dir, raw, vectorizer, suffix)

    else:
        corpus = [' '.join(corpus)]
        corpus_keywords('/data/share/liuchang/car_comment/mask/',corpus,vectorizer)
    




if __name__ == '__main__':
    main()