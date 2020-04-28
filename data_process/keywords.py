import pickle
from multiprocessing.pool import Pool

from data_process.utils import topn_words_from_dict, read_to_list, read_to_dict
from sklearn.feature_extraction.text import TfidfVectorizer
import os, re
import argparse
import numpy as np
from modification.change_mask import recover_mask
import time

from line_profiler import LineProfiler

class Keywords_Processor(object):
    def __init__(self, vectorizer=None,corpus=None,
                 stop_words:dict =None,
                 word_tfidf: dict = None,
                 corpus_keywords: dict = None,
                 max_features=None,
                 num_words=None,
                 **kwargs):
        if vectorizer:
            self.vectorizer = vectorizer
        elif corpus:
            self.vectorizer = TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)',
                                              stop_words=stop_words,
                                              max_features=max_features,
                                              **kwargs)
            self.vectorizer.fit(corpus)
        else:
            raise ValueError('Must provide either vectorizer or corpus.')
        
        self.feature_names = np.array(self.vectorizer.get_feature_names())

        self.stop_words = stop_words
        self.word_tfidf = word_tfidf
        self.corpus_keywords = corpus_keywords
        self.num_words=set(num_words) if num_words else set()
        print(type(self.num_words))
        assert isinstance(self.num_words,set)

    def extract_keywords(self, string, ratio=0.3):

        print(time.strftime("%H:%M:%S"))

        if not string:
            return ''

        if not self.corpus_keywords:
            self.corpus_keywords = {}

        words = string.split(' ')
        words = [w for w in words if not w.strip() in ',.?!:，。：！？、']

        words_set = set(words)
        stop_words_set = {w for w in words_set if w in self.stop_words}

        num = int(round(len(words_set) * ratio))
        num = max(1, num)

        keywords = []

        if self.corpus_keywords:
            keywords = [(word, self.corpus_keywords[word]) for word in words_set.intersection(self.corpus_keywords)]

            if len(keywords) > num:
                keywords.sort(key=lambda n: n[1], reverse=True)
                keywords = keywords[:num]

        keywords = [n[0] for n in keywords]

        if len(keywords) < num:
            tfidf = self.vectorizer.transform([string])

            zipped = list(zip(tfidf.data, tfidf.indices))
            zipped = [n for n in zipped if  self.feature_names[n[1]] in words_set-set(keywords)-stop_words_set]
            zipped.sort(key=lambda n: n[0], reverse=True)

            indexes = [n[1] for n in zipped[:num - len(keywords)]]
            keywords.extend(self.feature_names[indexes])

        if len(keywords) < num:
            remain = words_set - set(keywords) - stop_words_set
            if len(remain):
                keywords.extend(topn_words_from_dict(self.word_tfidf, num - len(keywords), remain))

        if len(keywords) < num:
            if len(stop_words_set):
                keywords.extend(topn_words_from_dict(self.word_tfidf, num - len(keywords), stop_words_set))

        keywords.extend([w for w in words_set.difference(keywords) if w in self.num_words])
        # 将关键词按它们在语句中出现的顺序排列
        keywords = [word for word in words if word in keywords]

        return ' '.join(keywords) + ' [sep] '

    def mask_unimportant_words(self, string, ratio=0.3,keep_nums=True):

        if not string:
            return ''

        print(time.strftime("%H:%M:%S"))
        if not self.corpus_keywords:
            self.corpus_keywords = {}

        feature_names = self.feature_names
        num_words=self.num_words
        assert isinstance(num_words,set)

        words = string.split(' ')
        words = [s for s in words if s != '']
        words_and_index = list(enumerate(words))

        words_and_index = [w for w in words_and_index if not w[1] in ',.?!:，。：！？、']

        if keep_nums:
            words_and_index=[w for w in words_and_index if not w[1] in num_words]

        num = int(round(len(words_and_index) * ratio))
        num = max(1, num)


        stop_words_index = [w for w in words_and_index if w[1] in self.stop_words]
        words_and_index = [w for w in words_and_index if not w[1] in self.stop_words]

        maskwords = []

        stop_tfidf={w:self.stop_words[w[1]] for w in stop_words_index}

        if len(stop_words_index):
            maskwords.extend(topn_words_from_dict(stop_tfidf, num,stop_words_index, reverse=False))

        if len(maskwords) < num:

            # remain = [w for w in words_and_index if not w[1] in self.stop_words and not w[1] in feature_names]
            #
            # if len(remain):
            #     maskwords.extend(topn_words_from_dict(word_tfidf, num - len(maskwords), remain))

            tfidf = self.vectorizer.transform([string])

            d = {feature_names[n[1]]: n[0] for n in zip(tfidf.data, tfidf.indices)}
            zipped = [(w, d[w[1]]) for w in words_and_index if w[1] in d and not w[1] in self.corpus_keywords]
            zipped.sort(key=lambda n: n[1])
            zipped = [n[0] for n in zipped[:num - len(maskwords)]]

            maskwords.extend(zipped)

            # 这段代码会选取关键词列表中tfidf值排名低的加入mask列表，注释后可以保留关键词，即使mask数量达不到ratio要求
            if  len(maskwords) < num:
                if self.corpus_keywords:
                    keywords = [(word, self.corpus_keywords[word[1]]) for word in words_and_index if
                                word[1] in self.corpus_keywords]

                    keywords.sort(key=lambda n: n[1])
                    keywords = [n[0] for n in keywords]
                    maskwords.extend(keywords[:num - len(maskwords)])

        maskindex = [n[0] for n in maskwords]
        words = ['<mask>' if i in maskindex else words[i] for i in range(len(words))]

        return ' '.join(words)

    def phrase_keywords_lists(self, texts, ratio=0.3):
        phrase_lists = [s.split('，') for s in texts]

        keywords_lists = [[self.extract_keywords(s, ratio=ratio) for s in l] for l in phrase_lists]

        return keywords_lists
    
    @staticmethod
    def corpus_tfidf( corpus,stop_words,out_dir):
        v=TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)')
        v.fit(corpus)
        
        strs=' '.join(corpus)
        
        tfidf = v.transform([strs]).toarray()[0]
        word_with_tfidf = list(zip(v.get_feature_names(), tfidf))
        word_with_tfidf.sort(key=lambda n: n[1], reverse=True)
        
        stops_tfidf=[]
        words_tfidf=[]
        for n in word_with_tfidf:
            if n[0] in stop_words:
                stops_tfidf.append(n)
            else:
                words_tfidf.append(n)

        # stops_tfidf=[n for n in word_with_tfidf if n[0] in stop_words]
        # word_tfidf=[n for n in word_with_tfidf if not n[0] in stop_words]
                
        stops_tfidf_str=[n[0]+'\t'+str(n[1]) for n in stops_tfidf]
        words_tfidf_str = [n[0]+'\t'+str(n[1]) for n in words_tfidf]

        with open(os.path.join(out_dir,'stop_words_tfidf'), 'w', encoding='utf8') as f:
            f.write('\n'.join(stops_tfidf_str))
        with open(os.path.join(out_dir,'words_tfidf'), 'w', encoding='utf8') as f:
            f.write('\n'.join(words_tfidf_str))

        num_words=[n for n in v.get_feature_names() if re.search(r'[0-9a-zA-Z]',n)]
        with open(os.path.join(out_dir,'num_words'), 'w', encoding='utf8') as f:
            f.write('\n'.join(num_words))

        return word_tfidf,stops_tfidf,num_words

def mask(s):
    global kp,args
    return kp.mask_unimportant_words(s,ratio=args.ratio)

def main():
    run = args.run
    if 'phrase' == run:
        for input_name in args.inputs:
            # recover_mask(data_dir,'test',suffix)

            # with open(os.path.join(data_dir, 'raw_test_corpus_{}'.format(suffix)), 'r', encoding='utf8') as f:
            #     raw = f.read().splitlines()
            with open(os.path.join(data_dir, input_name), 'r', encoding='utf8') as f:
                raw = f.read().splitlines()

            keywords_lists = kp.phrase_keywords_lists(raw, ratio=args.ratio)
            SEP = ' ||| '
            keywords = [SEP.join(l) for l in keywords_lists]

            with open(os.path.join(out_dir, '{}_phrase_keywords_lists_{}'.format(input_name, suffix)), 'w', encoding='utf8') as f:
                f.write('\n'.join(keywords))

    elif 'sentence' == run:
        for type in ['train', 'eval', 'test']:
            with open(os.path.join(data_dir, 'raw_{}_corpus_{}'.format(type, suffix)), 'r', encoding='utf8') as f:
                txts = f.read().splitlines()
            keywords = [kp.extract_keywords(s, ratio=args.ratio ) for
                        s in txts]

            with open(os.path.join(out_dir, '{}_keywords_{}_{}'.format(type, suffix, args.ratio)), 'w',
                      encoding='utf8') as f:
                f.write('\n'.join(keywords))

    elif 'mask' == run:
        process_num=args.process_num
        pool = Pool(process_num)

        for input_name in args.inputs:
            with open(os.path.join(data_dir, input_name), 'r', encoding='utf8') as f:
                raw = f.read().splitlines()

            prefix = [re.search(r'__.+?__ __.+?__ ', s) for s in raw]
            prefix=[n.group() if n else '' for n in prefix]
            # no_prefix = [raw[i].split(prefix[i])[-1].strip() for i in range(len(raw))]
            no_prefix = [r.split(p)[-1].strip() if p else r for r,p in zip(raw,prefix)]



            # kwd = dict(word_tfidf=kp.word_tfidf, corpus_keywords=kp.corpus_keywords, stop_words=kp.stop_words, ratio=args.ratio)
            # masked = [pool.apply_async(kp.mask_unimportant_words, args=(s,), kwds=kwd) for s in no_prefix]
            masked = pool.map(mask,no_prefix,chunksize=(len(no_prefix)+process_num-1)//process_num)
            # masked=[kp.mask_unimportant_words(s) for s in no_prefix]
            masked = [p + m for p, m in zip(prefix, masked)]

            with open(os.path.join(args.out_dir, input_name + '_masked_{}'.format(args.ratio)), 'w', encoding='utf8') as f:
                f.write('\n'.join(masked)+'\n')
            masked = [re.sub(r'(<mask> )+<mask>', '<mask>', s) for s in masked]
            with open(os.path.join(args.out_dir, input_name + '_infilling_masked_{}_keep_comma'.format(args.ratio)), 'w',
                      encoding='utf8') as f:
                f.write('\n'.join(masked)+'\n')
            masked = [re.sub(r'<mask>( <mask>| ，)+', '<mask>', s) for s in masked]
            with open(os.path.join(args.out_dir, input_name + '_infilling_masked_{}'.format(args.ratio)), 'w',
                      encoding='utf8') as f:
                f.write('\n'.join(masked)+'\n')

    else:
        corpus=read_to_list(args.corpus_path)
        word_with_tfidf = kp.corpus_tfidf(corpus,kp.stop_words,out_dir=args.out_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("run",
                        choices=["phrase", "corpus", "sentence", 'mask'],
                        help="Run type.")

    parser.add_argument("--data_dir",
                        default='/nfs/users/liuchang/comments_dayu',
                        help="The data directory.")
    parser.add_argument("--load_path",
                        default='/nfs/users/liuchang/comments_dayu/keywords_processor.pickle',
                        help="if given,load keywords_processor from the path.")
    parser.add_argument("--out_dir",
                        default='/nfs/users/liuchang/comments_dayu',
                        help="The output directory.")
    parser.add_argument("--inputs", nargs='+',
                        help="The input filenames.")
    parser.add_argument("--corpus_file",
                        default='/nfs/users/liuchang/comments_dayu/strs_10_80_p3_p10',
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
    parser.add_argument("--process_num", default=8, type=int,
                        help="process number of pool.")

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    load_path = args.load_path
    out_dir = args.out_dir
    suffix = args.suffix

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if os.path.isfile(load_path):
        with open(load_path, 'rb') as f:
            kp = pickle.load(f)

    else:
        corpus = read_to_list('/nfs/users/liuchang/comments_dayu/strs_10_80_p3_p10')

        stop_words = read_to_dict('/nfs/users/liuchang/comments_dayu/stop_words', '\t', float,None)
        corpus_keywords = read_to_dict('/nfs/users/liuchang/comments_dayu/words_tfidf', '\t', float, 1000)
        word_tfidf = read_to_dict('/nfs/users/liuchang/comments_dayu/words_tfidf', '\t', float, None)
        num_words = read_to_list('/nfs/users/liuchang/comments_dayu/num_words')

        kp = Keywords_Processor(corpus=corpus, stop_words=stop_words,
                                word_tfidf=word_tfidf,
                                corpus_keywords=corpus_keywords,
                                max_features=args.max_features,
                                num_words=num_words)
        with open(load_path, 'wb') as f:
            pickle.dump(kp,f)

    # lp=LineProfiler(main,kp.mask_unimportant_words)
    # lp.run('main()')
    # lp.print_stats()
    main()