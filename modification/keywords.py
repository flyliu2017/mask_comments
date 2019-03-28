from data_process.utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
import numpy as np
from modification.change_mask import recover_mask
import time


STOP_WORDS=read_to_list('/data/share/liuchang/car_comment/mask/stop_words')

class Keywords_Processor(object):
    def __init__(self,vectorizer=None,corpus=None,**kwargs):
        if vectorizer:
            self.vectorizer=vectorizer
        elif corpus:
            self.vectorizer=TfidfVectorizer(**kwargs)
            self.vectorizer.fit(corpus)
        else:
            raise ValueError('Must provide either vectorizer or corpus.')
        self.feature_names=np.array(self.vectorizer.get_feature_names())

    def extract_keywords(self, string, word_tfidf: dict = None, corpus_keywords: dict = None,
                         stop_words=None, ratio=0.3):
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

        words = string.split(' ')
        words = [w for w in words if w.strip() != '']

        words_set = set(words)
        stop_words_set = {w for w in words_set if w in stop_words}

        num = int(round(len(words_set) * ratio))
        num = max(1, num)

        keywords = []

        if corpus_keywords:
            keywords = [(word, corpus_keywords[word]) for word in words_set if word in corpus_keywords]

            if len(keywords) > num:
                keywords.sort(key=lambda n: n[1], reverse=True)
                keywords = keywords[:num]

        keywords = [n[0] for n in keywords]

        if len(keywords) < num:
            tfidf = self.vectorizer.transform([string])

            zipped = list(zip(tfidf.data, tfidf.indices))
            zipped = [n for n in zipped if not self.feature_names[n[1]] in keywords]
            zipped.sort(key=lambda n: n[0], reverse=True)

            indexes = [n[1] for n in zipped[:num - len(keywords)]]
            keywords.extend(self.feature_names[indexes])

        if len(keywords) < num:
            remain = words_set - set(keywords) - stop_words_set
            if len(remain):
                keywords.extend(topn_words_from_dict(word_tfidf, num - len(keywords), remain))

        if len(keywords) < num:
            if len(stop_words_set):
                keywords.extend(topn_words_from_dict(word_tfidf, num - len(keywords), stop_words_set))

        # 将关键词按它们在语句中出现的顺序排列
        keywords = [word for word in words if word in keywords]

        return ' '.join(keywords) + ' [sep] '

    def phrase_keywords_lists(self, texts):
        phrase_lists = [s.split('，') for s in texts]

        corpus_keywords = read_to_dict('/data/share/liuchang/car_comment/mask/corpus_keywords', '\t', float, 1000)
        word_tfidf = read_to_dict('/data/share/liuchang/car_comment/mask/word_tfidf', '\t', float, None)
        keywords_lists = [[self.extract_keywords( s,word_tfidf,corpus_keywords,stop_words=STOP_WORDS) for s in l] for l in phrase_lists]



        return keywords_lists

    def corpus_keywords(self, corpus):
        tfidf=self.vectorizer.transform(corpus).toarray()[0]
        word_with_tfidf=list(zip(self.feature_names,tfidf))
        word_with_tfidf.sort(key=lambda n:n[1],reverse=True)
        word_with_tfidf=[n[0]+'\t'+str(n[1]) for n in word_with_tfidf]

        return word_with_tfidf
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",
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
    
    vectorizer = TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)', stop_words=STOP_WORDS, max_features=args.max_features)
    vectorizer.fit(corpus)
    kp=Keywords_Processor(vectorizer)

    run=args.run
    if 'phrase'==run:
        recover_mask(data_dir,'test',suffix)
        
        with open(os.path.join(data_dir, 'raw_test_corpus_{}'.format(suffix)), 'r', encoding='utf8') as f:
            raw = f.read().splitlines()
            
        keywords_lists=kp.phrase_keywords_lists( raw)
        SEP = ' ||| '
        keywords = [SEP.join(l) for l in keywords_lists]

        with open(os.path.join(data_dir, 'phrase_keywords_lists_{}'.format(suffix)), 'w', encoding='utf8') as f:
            f.write('\n'.join(keywords))

    else:
        corpus = [' '.join(corpus)]
        word_with_tfidf=kp.corpus_keywords(corpus)

        with open('/data/share/liuchang/car_comment/mask/corpus_keywords', 'w', encoding='utf8') as f:
            f.write('\n'.join(word_with_tfidf))
    

if __name__ == '__main__':
    main()