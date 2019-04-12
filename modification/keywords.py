from data_process.utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
import numpy as np
from modification.change_mask import recover_mask
import time
import data_process.utils as utils




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
        if 'num_words' in kwargs:
            self.num_words=kwargs['num_words']
        else:
            self.num_words=[]

    def extract_keywords(self, string, word_tfidf: dict = None, corpus_keywords: dict = None,
                         stop_words=None, ratio=0.3):

        print(time.strftime("%H:%M:%S"))

        if not string:
            return ''

        if not corpus_keywords:
            corpus_keywords = {}

        words = string.split(' ')
        words = [w for w in words if not w.strip() in ',.?!:，。：！？、']

        words_set = set(words)
        stop_words_set = {w for w in words_set if w in stop_words}

        num = int(round(len(words_set) * ratio))
        num = max(1, num)

        keywords = []

        if corpus_keywords:
            keywords = [(word, corpus_keywords[word]) for word in words_set.intersection(corpus_keywords)]

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

        keywords.extend([w for w in words_set.difference(keywords) if w in self.num_words])
        # 将关键词按它们在语句中出现的顺序排列
        keywords = [word for word in words if word in keywords]

        return ' '.join(keywords) + ' [sep] '

    def mask_unimportant_words(self,string, word_tfidf: dict = None, corpus_keywords: dict = None,
                               stop_words=None, ratio=0.3):
        """

        :param vectorizer: tfidfvectorizer
        :param feature_names: words list from vectorizer.get_feature_names()
        :param string: string to be masked
        :param word_tfidf: a dict contain all words in corpus, { word : tfidf }
        :param corpus_keywords: the keywords list that we will search keywords in it firstly.
        :param stop_words: stop words
        :param ratio: the ratio of mask.
        :return:
        """

        if not string:
            return ''

        print(time.strftime("%H:%M:%S"))
        if not corpus_keywords:
            corpus_keywords={}

        feature_names = np.array(self.feature_names)

        words = string.split(' ')
        words_and_index=list(enumerate(words))

        words_and_index=[w for w in words_and_index if not w[1] in ',.?!:，。：！？、']

        stop_words_index = [w for w in words_and_index if w[1] in stop_words]

        num = int(round(len(words_and_index) * ratio))
        num = max(1, num)

        maskwords = []

        word_tfidf={ w:word_tfidf[w[1]] for w in words_and_index}

        if len(stop_words_index):
            maskwords.extend(topn_words_from_dict(word_tfidf, num, stop_words_index ,reverse=False))

        if len(maskwords) < num:

            remain = [ w for w in words_and_index if not w[1] in stop_words and not w[1] in feature_names ]

            if len(remain):
                maskwords.extend(topn_words_from_dict(word_tfidf, num-len(maskwords), remain))

            if len(maskwords) < num:
                tfidf = self.vectorizer.transform([string])

                d = {feature_names[n[1]]:n[0] for n in  zip(tfidf.data, tfidf.indices)}
                zipped=[(w,d[w[1]]) for w in words_and_index if w in d and not w in corpus_keywords]
                zipped.sort(key=lambda n: n[1])
                zipped = [n[0] for n in zipped[:num - len(maskwords)]]

                maskwords.extend(zipped)

                # 这段代码会选取关键词列表中tfidf值排名低的加入mask列表，注释后可以保留关键词，即使mask数量达不到ratio要求

                # if len(maskwords) < num:
                #     if corpus_keywords:
                #         keywords = [(word, corpus_keywords[word]) for word in words_and_index if word in corpus_keywords]
                #
                #         keywords.sort(key=lambda n: n[1])
                #         maskwords.extend(keywords[:num-len(maskwords)])

        maskindex = [n[0] for n in maskwords]
        words=['<mask>' if i in maskindex else words[i] for i in range(len(words))]

        return ' '.join(words)

    def phrase_keywords_lists(self, texts,word_tfidf,corpus_keywords):
        phrase_lists = [s.split('，') for s in texts]


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
                        choices=["phrase", "corpus","sentence",'mask'],
                        help="Run type.")

    parser.add_argument("--data_dir",
                        default='/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask',
                        help="The data directory.")
    parser.add_argument("--input",
                        default='raw_test_corpus_only_mask',
                        help="The input filename.")
    parser.add_argument("--corpus_file",
                        default='/data/share/liuchang/car_comment/mask/selected_str_5_80_p5_p10.txt',
                        help="The corpus file.")
    parser.add_argument("--suffix",
                        default='',
                        help="The output filename suffix.")
    parser.add_argument("--ratio",
                        type=float,
                        default=0.3,
                        help="The keywords ratio.")

    parser.add_argument("--max_features", default=50000, type=int,
                        help="max_features for tfidfvectorizer.")

    args = parser.parse_args()
    print(args)

    data_dir=args.data_dir
    suffix = args.suffix

    corpus = read_to_list(args.corpus_file)

    STOP_WORDS = read_to_list('/data/share/liuchang/car_comment/mask/stop_words')
    corpus_keywords = read_to_dict('/data/share/liuchang/car_comment/mask/corpus_keywords', '\t', float, 1000)
    word_tfidf = read_to_dict('/data/share/liuchang/car_comment/mask/word_tfidf', '\t', float, None)
    num_words=read_to_list('/data/share/liuchang/car_comment/mask/mask_comments/data_process/key_words_all.txt')

    vectorizer = TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)', stop_words=STOP_WORDS, max_features=args.max_features)
    vectorizer.fit(corpus)
    kp=Keywords_Processor(vectorizer,num_words=num_words)

    run=args.run
    if 'phrase'==run:
        # recover_mask(data_dir,'test',suffix)
        
        # with open(os.path.join(data_dir, 'raw_test_corpus_{}'.format(suffix)), 'r', encoding='utf8') as f:
        #     raw = f.read().splitlines()
        with open(os.path.join(data_dir, args.input), 'r', encoding='utf8') as f:
            raw = f.read().splitlines()

        keywords_lists=kp.phrase_keywords_lists( raw,corpus_keywords=corpus_keywords,word_tfidf=word_tfidf)
        SEP = ' ||| '
        keywords = [SEP.join(l) for l in keywords_lists]

        with open(os.path.join(data_dir, 'phrase_keywords_lists_{}'.format(suffix)), 'w', encoding='utf8') as f:
            f.write('\n'.join(keywords))

    elif 'sentence'==run:
        for type in ['train','eval','test']:
            with open(os.path.join(data_dir, 'raw_{}_corpus_{}'.format(type, suffix)), 'r', encoding='utf8') as f:
                txts = f.read().splitlines()
            keywords = [kp.extract_keywords(s, word_tfidf, corpus_keywords, stop_words=STOP_WORDS,ratio=args.ratio) for s in txts]

            with open(os.path.join(data_dir, '{}_keywords_{}_{}'.format(type, suffix,args.ratio)), 'w', encoding='utf8') as f:
                f.write('\n'.join(keywords))

    elif 'mask'==run:
        with open(os.path.join(data_dir, args.input), 'r', encoding='utf8') as f:
            raw = f.read().splitlines()

        masked = [kp.mask_unimportant_words(s, corpus_keywords=corpus_keywords, word_tfidf=word_tfidf) for s in raw]

        with open(os.path.join(data_dir, args.input+'_masked_{}'.format(args.ratio)), 'w', encoding='utf8') as f:
            f.write('\n'.join(masked))

    else:
        corpus = [' '.join(corpus)]
        word_with_tfidf=kp.corpus_keywords(corpus)

        with open('/data/share/liuchang/car_comment/mask/corpus_keywords', 'w', encoding='utf8') as f:
            f.write('\n'.join(word_with_tfidf))
    

if __name__ == '__main__':
    main()