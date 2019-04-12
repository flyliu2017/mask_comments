# coding=utf8
import argparse
import jieba
import json
import os
import re
import yaml

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from data_process.utils import *
from modification.keywords import Keywords_Processor
import time


STOP_WORDS=read_to_list('/data/share/liuchang/car_comment/mask/stop_words')

class Processor(object):
    def __init__(self, out_dir, df_path=None, raw_path=None, save_path='df.json'):
        if not df_path:
            if not raw_path:
                raise ValueError('Must provide df or raw.')
            else:
                self.segment(raw_path, save_path)
        else:
            self.df = pd.read_json(df_path, orient='index')
            self.label_list = self.df.columns[3:]
            self.comments_str=self.df[self.label_list]

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.out_dir = out_dir
        self.count=0

    def segment(self, raw_path, output):
        raw = []
        with open(raw_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                raw.append(json.loads(line))

        df = pd.DataFrame([raw[i].values() for i in range(len(raw))],
                          columns=raw[0].keys())

        self.label_list = df.columns[3:]

        comments = df[self.label_list]
        join = lambda s: ''.join(s.split(' '))
        comments_joined = comments.applymap(join)

        comments_list = comments_joined.applymap(lambda s: list(jieba.cut(s)))
        comments_list = comments_list.applymap(lambda l: [w for w in l if w.strip() != ''])

        self.comments_str = comments_list.applymap(lambda s: ' '.join(s))

        df[self.label_list] = self.comments_str
        df.to_json(output, force_ascii=False, orient='index')

        self.df = df

    @staticmethod
    def mask_phrase_str(string, mask_index=None, mode='random', min_index=0, max_index=128):
        l = re.split('[，。]', re.sub('[!?;！？；]|… …|…', '，', string))
        l = [s for s in l if s.strip() != '']
        if not l:
            return ''

        if not mask_index:
            if mode == 'random':
                mask_index = np.random.randint(min_index, min(len(l), max_index))
            elif mode == 'median':
                mask_index = len(l) // 2
            else:
                raise ValueError('mode should be "random" or "median"')

        label = l[mask_index]
        corpus = '，'.join(l[:mask_index] + [' <mask> '] + l[mask_index + 1:])

        return corpus + '<separate>' + label

    @staticmethod
    def mask_for_context(string):
        l = re.split('[，。]', re.sub('[!?;！？；]|… …|…', '，', string))
        l = [s for s in l if s.strip() != '']
        if not l:
            return ''

        context = '，'.join([l[0], ' <mask> ', l[-1]])
        label = '，'.join(l[1:-1])

        return context + '<separate>' + label

    def generate_corpus(self, df, suffix=''):
        l = df.values.flatten()
        l = [s for s in l if s != '']

        with open(os.path.join(self.out_dir, 'corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
            f.write('\n'.join(l))

        df = self.add_prefix(df)
        l = df.values.flatten()
        l = [s for s in l if '，' in s]

        with open(os.path.join(self.out_dir, 'prefixed_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
            f.write('\n'.join(l))

    def add_prefix(self, dataframe, add_class=True, add_types=True):
        if not add_class and not add_types:
            return

        prefixed_df = dataframe.copy()

        for l in prefixed_df.columns:
            if add_class:
                prefixed_df[l] = pd.Series(['__' + l + '__ '] * len(prefixed_df)).str.cat(
                    prefixed_df[l])
            if add_types:
                types = self.df['question_forum'].map(lambda s: '__' + s + '__ ')
                prefixed_df[l] = types.str.cat(prefixed_df[l])

        return prefixed_df

    # def init_vectorizer(self):
    #
    #     self.vectorizer = TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)', stop_words=STOP_WORDS,max_features=30000)
    #     self.vectorizer.fit(self.comments_str.values.flatten())
    #     self.feature_names = np.array(self.vectorizer.get_feature_names())

    def generate_yaml(self, suffix):
        d = {'model_dir': 'transformer_' + suffix,
             'data':
                 {'train_features_file': 'train_corpus_{}'.format(suffix),
                  'train_labels_file': 'train_labels_{}'.format(suffix),
                  'eval_features_file': 'eval_corpus_{}'.format(suffix),
                  'eval_labels_file': 'eval_labels_{}'.format(suffix),
                  'source_words_vocabulary': 'vocab_{}'.format(suffix),
                  'target_words_vocabulary': 'vocab_{}'.format(suffix)
                  },
             'train': {'train_steps': 100000},
             'eval' : {'eval_delay' : 1800 },
             'params': {'decay_params':{'warmup_steps' : 8000 },
                        'learning_rate': 1.0}
             }
        with open(os.path.join(self.out_dir, 'train.yml'), 'w', encoding='utf8') as f:
            f.write(yaml.dump(d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",
                        choices=["mask", "compare_mask","compare_result", "bleu", "context","mask_unimportant"],
                        help="Run type.")

    parser.add_argument("--df_path",
                        default='/data/share/liuchang/car_comment/mask/comments_5_80_p5_p10.json',
                        help="The path to dataframe file. If file not exist, dataframe will be saved at this location.")
    parser.add_argument("--raw_path",
                        help="The path to raw data file.")
    parser.add_argument("--output_dir", default='',
                        help="The output directory.")
    parser.add_argument("--class_str", default='',
                        help="The class of comments need to be extracted.")
    parser.add_argument("--suffix", default='',
                        help="The suffix of output file names.")
    parser.add_argument("--preds_path",
                        help="Path to prediction file.")
    parser.add_argument("--corpus_path",
                        help="Path to corpus file.")
    parser.add_argument("--labels_path",
                        help="Path to labels file.")
    parser.add_argument("--compare_path", default='compare',
                        help="The name of result-comparing file.")
    parser.add_argument("--bleu_path", default='bleu',
                        help="The name of bleu file.")

    parser.add_argument("--min_words", default=5, type=int,
                        help="Minimal word number of input string.")
    parser.add_argument("--max_words", default=40, type=int,
                        help="Maximal word number of input string.")
    parser.add_argument("--min_phrase", default=5, type=int,
                        help="Minimal phrase number of input string.")
    parser.add_argument("--max_phrase", default=10, type=int,
                        help="Maximal phrase number of input string.")
    parser.add_argument("--add_prefix", default=True, type=bool,
                        help="Add prefix or not.")
    parser.add_argument("--add_keywords", default='', choices=['whole','only_mask',''],
                        help="Choose keywords. 'whole' for sentence, 'only_mask' for masked phrase, '' for no keywords.")
    parser.add_argument("--mask_index", default=None, type=int,
                        help="Index of the phrase to be masked.")
    parser.add_argument("--ratio", default=0.3, type=float,
                        help="Ratio of words to be masked.")
    parser.add_argument("--mode", default="random", choices=["random", "median"],
                        help="mask mode.Ignored when mask_index is not None.")

    args = parser.parse_args()
    print(args)

    suffix=args.suffix
    output_dir=args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if args.run == 'mask' or args.run=='context' or args.run=='mask_unimportant':
        p = Processor(output_dir, args.df_path)
        p.generate_yaml(suffix)
        df=p.comments_str
        if args.class_str:
            df=pd.DataFrame(df[args.class_str])
        # df=df.applymap(lambda s: p.length_selection(s,args.min_words, args.max_words))
        # df=df.applymap(lambda s: p.phrase_selection(s,args.min_phrase, args.max_phrase))

        selected_strings = df.copy()

        corpus = read_to_list('/data/share/liuchang/car_comment/mask/selected_str_5_80_p5_p10.txt')
        STOP_WORDS = read_to_list('/data/share/liuchang/car_comment/mask/stop_words')
        corpus_keywords = read_to_dict('/data/share/liuchang/car_comment/mask/corpus_keywords', '\t', float, 1000)
        word_tfidf = read_to_dict('/data/share/liuchang/car_comment/mask/word_tfidf', '\t', float, None)
        num_words = read_to_list('/data/share/liuchang/car_comment/mask/mask_comments/data_process/key_words_all.txt')

        vectorizer = TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)', stop_words=STOP_WORDS,
                                     max_features=args.max_features)
        vectorizer.fit(corpus)
        kp = Keywords_Processor(vectorizer, num_words=num_words)

        if args.run=='mask_unimportant':

            df=df.applymap(lambda s: kp.mask_unimportant_words(s,  word_tfidf=word_tfidf,
                                                           corpus_keywords=corpus_keywords, stop_words=STOP_WORDS,ratio=args.ratio))

        else:
            f=p.mask_phrase_str if args.run=='mask' else p.mask_for_context
            df=df.applymap(lambda s:f(s,mask_index=args.mask_index,mode=args.mode))

        if args.add_prefix:
            df=p.add_prefix(df)

        strings = df.values.flatten()
        strings = [s.split('<separate>') for s in strings]
        strings = [n for n in strings if len(n) == 2]

        corpus ,labels=zip(*strings)

        if args.run!='mask_unimportant' and ''!=args.add_keywords:

            if 'whole'==args.add_keywords:

                kw_strings=selected_strings.values.flatten()
                kw_strings=[ s for s in kw_strings if s !='' ]
            else:
                kw_strings=labels

            keywords = [ kp.extract_keywords(s,word_tfidf=word_tfidf,
                                                corpus_keywords=corpus_keywords,stop_words=STOP_WORDS)
                         for s in kw_strings]

            z=zip(keywords,corpus)
            corpus=[ n[0]+ n[1] for n in z]

        slice_ratios = [0.9, 0.09]
        generate_dataset(corpus, labels, output_dir, slice_ratios, suffix)



    elif args.run == 'compare_mask':

        p, l, c = map(read_to_list, [args.preds_path, args.labels_path, args.corpus_path])

        compare_mask(p, l, c, os.path.join(output_dir, args.compare_path))

    elif args.run == 'compare_result':
        p, l, c = map(read_to_list, [args.preds_path, args.labels_path, args.corpus_path])

        compare_result(p, l, c, os.path.join(output_dir, args.compare_path))

    elif args.run == 'bleu':
        file_bleu(args.preds_path,
                            args.labels_path,
                            os.path.join(output_dir, args.bleu_path))

    elif args.run == 'corpus':
        p = Processor(output_dir, args.df_path)
        df = p.comments_str.applymap(lambda s: p.length_selection(s, args.min_words, args.max_words))
        df = df.applymap(lambda s: p.phrase_selction(s, args.min_phrase, args.max_phrase))
        p.generate_corpus(df, suffix)


def generate_dataset(corpus, labels, output_dir, slice_ratios, suffix):
    with open(os.path.join(output_dir, 'masked_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(corpus))

    with open(os.path.join(output_dir, 'labels_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(labels))

    shuffle_index = np.random.permutation(list(range(len(corpus))))
    paths = [os.path.join(output_dir, '{}_corpus_{}'.format(s, suffix)) for s in ['train', 'eval', 'test']]
    slice_and_save(corpus, shuffle_index, slice_ratios, paths)
    paths = [os.path.join(output_dir, '{}_labels_{}'.format(s, suffix)) for s in ['train', 'eval', 'test']]
    slice_and_save(labels, shuffle_index, slice_ratios, paths)


if __name__ == "__main__":
    main()
