from data_process.processor import Processor
from modification.change_mask import recover_mask
from data_process.utils import extract_keywords
from sklearn.feature_extraction.text import TfidfVectorizer
import os


def main():
    out_dir = '/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask'
    suffix = 'only_mask'
    recover_mask(out_dir, suffix)
    p = Processor(out_dir, '/data/share/liuchang/car_comment/mask/df.json')
    p.init_vectorizer()

    with open(os.path.join(out_dir, 'no_prefix_test_corpus_{}'.format(suffix)), 'r', encoding='utf8') as f:
        corpus = f.read().splitlines()

    phrase_lists = [s.split('，') for s in corpus]
    keywords_lists = [[extract_keywords(p.vectorizer, p.feature_names, s) for s in l] for l in phrase_lists]
    SEP=' ||| '
    keywords=[SEP.join(l) for l in keywords_lists]

    with open(os.path.join(out_dir,'keywords_lists_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(keywords))
    
if __name__ == '__main__':
    main()