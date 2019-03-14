# coding=utf8
import sacrebleu
import argparse
import os
import json, jieba, yaml
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time

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
        self.vectorizer: TfidfVectorizer = None
        self.feature_names=None
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
    def length_selection(string, min_words=5, max_words=40):
        l = string.split(' ')
        l = [s for s in l if s.strip() != '']
        l = l if min_words <= len(l) <= max_words else []
        return ' '.join(l)

    @staticmethod
    def phrase_selction(string, min_phrase=5, max_phrase=10):
        l = re.split('[，。]', re.sub('[!?;！？；]|… …|…', '，', string))
        l = [s for s in l if s.strip() != '']
        if len(l) < min_phrase or len(l) > max_phrase:
            return ''
        else:
            return '，'.join(l)

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

    def extract_keywords(self,string,ratio=0.3):
        self.count+=1
        if self.count%100==0:
            print(time.strftime("%H:%M:%S")+' '+ str(self.count))
        if not string:
            return ''


        tfidf = self.vectorizer.transform([string])


        num=int(len(string.split(' '))*ratio)
        z = list(zip(tfidf.data, tfidf.indices))
        z.sort(key=lambda n: n[0], reverse=True)
        indexes = [n[1] for n in z[:num]]

        return ' '.join(self.feature_names[indexes]) + ' [sep] '



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

    @staticmethod
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

    @staticmethod
    def compare_result(preds, labels, corpus, output='compare'):
        if len(preds) != len(labels) or len(labels) != len(corpus):
            raise ValueError("predictions,labels and corpus should have same length.")

        corpus=[s.split('[sep]')[-1] for s in corpus]
        result = []
        for p, l, c in zip(preds, labels, corpus):
            result.append(c.replace('mask', l + " | " + p))
        with open(output, 'w', encoding='utf8') as f:
            f.write('\n'.join(result))

    @staticmethod
    def read(path):
        with open(path, 'r') as f:
            l = f.read().splitlines()
        return l

    @staticmethod
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

    @staticmethod
    def file_bleu(pred_path, labels_path, output="bleu"):
        with open(pred_path, 'r', encoding='utf8') as f:
            predictions = f.readlines()
        with open(labels_path, 'r', encoding='utf8') as f:
            labels = f.readlines()
        Processor.cal_bleu(predictions, labels, output)

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
             'eval' : {'eval_delay' : 3600 }
             }
        with open(os.path.join(self.out_dir, 'train.yml'), 'w', encoding='utf8') as f:
            f.write(yaml.dump(d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",
                        choices=["mask", "compare", "bleu", "context"],
                        help="Run type.")

    parser.add_argument("--df_path",
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
    parser.add_argument("--add_keywords", default='whole', choices=['whole','only_mask',''],
                        help="Choose keywords. 'whole' for sentence, 'only_mask' for masked phrase, '' for no keywords.")
    parser.add_argument("--mask_index", default=None, type=int,
                        help="Index of the phrase to be masked.")
    parser.add_argument("--mode", default="random", choices=["random", "median"],
                        help="mask mode.Ignored when mask_index is not None.")

    args = parser.parse_args()
    print(args)

    suffix=args.suffix
    output_dir=args.output_dir

    if args.run == 'mask' or args.run=='context':
        p = Processor(output_dir, args.df_path)
        p.generate_yaml(suffix)
        df=p.comments_str
        if args.class_str:
            df=pd.DataFrame(df[args.class_str])
        df=df.applymap(lambda s: p.length_selection(s,args.min_words, args.max_words))
        df=df.applymap(lambda s: p.phrase_selction(s,args.min_phrase, args.max_phrase))

        selected_strings=df.copy()

        f=p.mask_phrase_str if args.run=='mask' else p.mask_for_context
        df=df.applymap(lambda s:f(s,mask_index=args.mask_index,mode=args.mode))

        if args.add_prefix:
            df=p.add_prefix(df)

        strings = df.values.flatten()
        strings = [s.split('<separate>') for s in strings]
        strings = [n for n in strings if len(n) == 2]

        corpus ,labels=zip(*strings)

        if ''!=args.add_keywords:
            if not p.vectorizer:
                p.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.7,max_features=30000)
                p.vectorizer.fit(p.df.values.flatten())
            p.feature_names = np.array(p.vectorizer.get_feature_names())


            if 'whole'==args.add_keywords:

                kw_strings=selected_strings.values.flatten()
                kw_strings=[s for s in kw_strings if s !='']

            else:
                kw_strings=labels

            keywords = map(p.extract_keywords,kw_strings)

            z=zip(keywords,corpus)
            corpus=[ n[0]+ n[1] for n in z]


        with open(os.path.join(output_dir, 'masked_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
            f.write('\n'.join(corpus))

        with open(os.path.join(output_dir, 'labels_{}'.format(suffix)), 'w', encoding='utf8') as f:
            f.write('\n'.join(labels))

        slice_ratios = [0.8, 0.1]
        shuffle_index = np.random.permutation(list(range(len(corpus))))

        paths = [os.path.join(output_dir, '{}_corpus_{}'.format(s, suffix)) for s in ['train', 'eval', 'test']]
        Processor.slice_and_save(corpus, shuffle_index, slice_ratios, paths)

        paths = [os.path.join(output_dir, '{}_labels_{}'.format(s, suffix)) for s in ['train', 'eval', 'test']]
        Processor.slice_and_save(labels, shuffle_index, slice_ratios, paths)



    elif args.run == 'compare':

        p, strings, c = map(Processor.read, [args.preds_path, args.labels_path, args.corpus_path])

        Processor.compare_result(p, strings, c, os.path.join(output_dir , args.compare_path))

    elif args.run == 'bleu':
        Processor.file_bleu(args.preds_path,
                            args.labels_path,
                            os.path.join(output_dir, args.bleu_path))

    elif args.run == 'corpus':
        p = Processor(output_dir, args.df_path)
        df = p.comments_str.applymap(lambda s: p.length_selection(s, args.min_words, args.max_words))
        df = df.applymap(lambda s: p.phrase_selction(s, args.min_phrase, args.max_phrase))
        p.generate_corpus(df, suffix)


if __name__ == "__main__":
    main()
