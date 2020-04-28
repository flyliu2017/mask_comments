import json
import multiprocessing
import os
import re
import time
from multiprocessing.pool import Pool

import jieba as jieba
from simplex_sdk import SimplexClient
import tensorflow as tf
import numpy as np
import subprocess
from cal_scores import CalScore

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", '/nfs/users/liuchang/comments_dayu/tag_prediction/data',
    "The data dir.")
flags.DEFINE_string(
    "input_file", 'test_xs_multitags',
    "The input file. ")
flags.DEFINE_string(
    "output_file", 'test_pred_tags',
    "The output file. ")
flags.DEFINE_string(
    "device", '3',
    "CUDA_VISIBLE_DEVICES ")
flags.DEFINE_integer(
    "rewrite_num", None,
    "number of inputs to rewrite.")
flags.DEFINE_enum('mode','tag_and_mark',
                  enum_values=['tag_to_mark','tag_and_mark','mark_to_tag'],
                  help="rewrite mode.")

def get_tags(inputs,batch_size=10):
    client = SimplexClient('BertCarMultiLabelsExtractTopK')
    results=[]
    iter=(len(inputs)+batch_size-1)//batch_size
    for i in range(iter):
        data=[{"content":n} for n in inputs[i*batch_size:(i+1)*batch_size]]
        ret = client.predict(data)
        print(ret)
        ret=[n['tags'] for n in ret]
        ret=[[n['tag'] for n in l] for l in ret]
        results.extend(ret)
    return results


def extract_phrases_based_on_tags(combined_input):

    combined_file = os.path.join(FLAGS.data_dir, 'temp/combined_' + FLAGS.output_file)
    with open(combined_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(combined_input))

    cmd = '/opt/anaconda/anaconda3/envs/tf1.12/bin/python run_token_level_classifier_from_file.py    ' \
          '--task_name=fromfile    ' \
          '--do_predict=true   ' \
          '--data_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/data   ' \
          '--vocab_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/vocab.txt    ' \
          '--bert_config_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_config.json    ' \
          '--init_checkpoint=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_model.ckpt    ' \
          '--max_seq_length=128    ' \
          '--train_batch_size=32    ' \
          '--learning_rate=2e-5    ' \
          '--num_train_epochs=6.0 ' \
          '--output_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/rewrite/ ' \
          '--from_file {} '.format(combined_file)
    ret=subprocess.run(cmd,shell=True)
    if ret.returncode:
        raise SystemError('extract_phrases_based_on_tags failed.')

    with open(os.path.join('/nfs/users/liuchang/comments_dayu/tag_prediction/rewrite/test_results.tsv'), 'r',
              encoding='utf8') as f:
        marks = f.read().splitlines()
    return marks

def extract_phrases_and_tags(inputs):

    os.chdir('/nfs/users/liuchang/bert')

    def batch_inputs(inputs,batch_size=10000):
        num=(len(inputs)+batch_size-1)//batch_size

        for i in range(num):
            yield inputs[i*batch_size:(i+1)*batch_size]

    marks_and_tags=[]
    temp_file = os.path.join(FLAGS.data_dir, 'temp.txt')

    for input in batch_inputs(inputs):
        with open(temp_file, 'w', encoding='utf8') as f:
            txts = [ n+'\n' for n in input ]
            f.writelines(txts)

        cmd = '/opt/anaconda/anaconda3/envs/tf1.12/bin/python -m bin.run    ' \
              '--task_name=phrase-and-tag    ' \
              '--do_predict=true   ' \
              '--data_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/data   ' \
              '--vocab_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/vocab.txt    ' \
              '--bert_config_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_config.json    ' \
              '--init_checkpoint=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_model.ckpt    ' \
              '--max_seq_length=128    ' \
              '--train_batch_size=32    ' \
              '--learning_rate=2e-5    ' \
              '--num_train_epochs=6.0 ' \
              '--output_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/bert_extract_phrase_and_tag/ ' \
              '--data_converted=false ' \
              '--predict_from_file {} '.format(temp_file)

        ret = subprocess.run(cmd, shell=True)
        if ret.returncode:
            raise SystemError('predict_tags_from_phrases failed.')

        with open('/nfs/users/liuchang/comments_dayu/tag_prediction/bert_extract_phrase_and_tag/predict_result.tsv', 'r',
                  encoding='utf8') as f:
            results = f.readlines()
            marks_and_tags.extend([n.strip() for n in results])
    return marks_and_tags

def extract_phrases(raws):
    raise NotImplementedError

def predict_tags_from_phrases(phrases):

    # input_file = os.path.join(data_dir, 'temp/phrases')
    # with open(input_file, 'w', encoding='utf8') as f:
    #     f.write('\n'.join(phrases))
    #
    # os.chdir('/nfs/users/liuchang/bert')
    # cmd = '/opt/anaconda/anaconda3/envs/tf1.12/bin/python -m bin.run    ' \
    #       '--task_name=multitag    ' \
    #       '--do_predict=true   ' \
    #       '--data_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/data   ' \
    #       '--vocab_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/vocab.txt    ' \
    #       '--bert_config_file=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_config.json    ' \
    #       '--init_checkpoint=/nfs/users/ludezheng/bert/chinese_L-12_H-768_A-12/bert_model.ckpt    ' \
    #       '--max_seq_length=128    ' \
    #       '--train_batch_size=32    ' \
    #       '--learning_rate=2e-5    ' \
    #       '--num_train_epochs=6.0 ' \
    #       '--output_dir=/nfs/users/liuchang/comments_dayu/tag_prediction/bert_phrase_tag/ ' \
    #       '--data_converted=false ' \
    #       '--predict_from_file {} '.format(input_file)
    # ret = subprocess.run(cmd, shell=True)
    # if ret.returncode:
    #     raise SystemError('predict_tags_from_phrases failed.')

    with open('/nfs/users/liuchang/comments_dayu/tag_prediction/bert_phrase_tag/predict_result.tsv', 'r', encoding='utf8') as f:
        flatten_tags = f.read().splitlines()
    flatten_tags=[n.split('\t')[0] for n in flatten_tags]
    return flatten_tags

def get_candidates(phrases_bag:dict,tags,old_phrases,candidate_num=5):
    candidate_bags={}
    for phrase,tag in zip(old_phrases,tags):
        if phrase.strip()=='' or not tag in phrases_bag:
            continue
        if phrase in candidate_bags:
            candidate_bags[phrase]=candidate_bags[phrase]+phrases_bag[tag]
        else:
            candidate_bags[phrase]=phrases_bag[tag]

    for phrase in candidate_bags:
        bag=candidate_bags[phrase]
        if len(bag)>candidate_num:
            candidate_bags[phrase]=np.random.choice(bag,candidate_num)

    return candidate_bags

def generate_all_rewrite_candidates(sentence,candidate_bags):
    if not candidate_bags:
        return [sentence.lower()]
    keys=list(candidate_bags.keys())
    semi_rewrites=generate_all_rewrite_candidates(sentence,{key:candidate_bags[key] for key in keys[1:]})
    key=keys[0]
    rewrites=[semi_rewrite.replace(key,'***' + s + '***') for s in candidate_bags[key] for semi_rewrite in semi_rewrites]
    return rewrites

# @profile
def choose_rewrite_by_ppl(lm,sentence,candidate_bags,strategy='beamsearch'):
    # @profile
    def choose_rewrite(sentence,candidate_bags):
        candidates = generate_all_rewrite_candidates(sentence, candidate_bags)
        seged_candidates = [' '.join(list(jieba.cut(n.replace('***', '')))) for n in candidates]
        ppls = lm.get_ppl_from_lm(seged_candidates)
        sentence = candidates[ppls.index(min(ppls))]
        return sentence

    if strategy=='beamsearch':
        return choose_rewrite(sentence,candidate_bags)
    # elif strategy=='greedy':
    else:
        for key in candidate_bags:
            sentence=choose_rewrite(sentence,{key:candidate_bags[key]})

        return sentence



def marks_to_phrases(marks,tags,sentence):
    old_phrases = re.split(r'[，。；！？]', sentence.lower())
    old_phrases = [n for n in old_phrases if n.strip() != '']

    phrases = []
    new_tags=[]
    for mark,tag in zip(marks,tags):
        if mark=='':
            continue
        if '，' in mark:
            phrases.append(mark)
            new_tags.append(tag)
        else:
            for phrase in old_phrases:
                if mark in phrase:
                    phrases.append(phrase)
                    new_tags.append(tag)
                    break

    return phrases,new_tags

# @profile
def rewrite(phrases_bag:dict,sentence,tags,old_phrases,language_model:CalScore,candidate_num=5,strategy='beamsearch'):
    candidate_bags=get_candidates(phrases_bag,tags,old_phrases,candidate_num)
    if not candidate_bags:
        return sentence

    if strategy in ['beamsearch','greedy']:
        return choose_rewrite_by_ppl(language_model,sentence,candidate_bags,strategy=strategy)

    if strategy=='random':
        for phrase in candidate_bags:
            new=np.random.choice(candidate_bags[phrase])
            sentence = sentence.replace(phrase, '***' + new + '***')
        return sentence

    raise ValueError('strategy must be one of "beamsearch,greedy,random".')

def tag_to_mark(raws):

    output_file = os.path.join(FLAGS.data_dir, FLAGS.output_file)

    if not os.path.isfile(output_file):
        pred_tags = get_tags(raws)
        outstr = [' | '.join(n) for n in pred_tags]
        print(pred_tags)
        with open(os.path.join(FLAGS.data_dir, FLAGS.output_file), 'w', encoding='utf8') as f:
            f.write('\n'.join(outstr))
    else:
        with open(os.path.join(FLAGS.data_dir, FLAGS.output_file), 'r', encoding='utf8') as f:
            outstr = f.read().splitlines()
            pred_tags = [n.split(' | ') for n in outstr]

    lengths = [len(n) for n in pred_tags]
    repeat_inputs = []
    for i in range(len(raws)):
        repeat_inputs.extend([raws[i]] * lengths[i])

    flatten = [n for l in pred_tags for n in l]
    combine = [i + ' | ' + tag for i, tag in zip(repeat_inputs, flatten)]

    flatten_marks=extract_phrases_based_on_tags(combine)

    lengths=[len(n) for n in pred_tags]
    cum_len = [0] + np.cumsum(lengths).tolist()
    marks=[flatten_marks[cum_len[i]:cum_len[i+1]] for i in range(len(cum_len)-1)]

    return pred_tags,marks

# @profile
def tag_and_mark(raws):


    marks_and_tags=extract_phrases_and_tags(raws)

    marks_and_tags=[n.split('\t') for n in marks_and_tags]
    marks_and_tags=[[n.split(': ') for n in l if n!=''] if l!=[''] else [['','']] for l in marks_and_tags]
    marks_and_tags=[list(zip(*n)) for n in marks_and_tags]
    tags,marks =list(zip(*marks_and_tags))

    return tags, marks


def mark_to_tag(raws):

    with open('/nfs/users/liuchang/comments_dayu/tag_prediction/extract_all_phrase/test_results.tsv', 'r',
              encoding='utf8') as f:
        marks = f.read().splitlines()
    marks=[mark.split() for mark in marks]
    marks=[[n for n in l if len(n)>2] for l in marks]
    lengths=[len(n) for n in marks]
    flatten=[n for l in marks for n in l]

    flatten_tags=predict_tags_from_phrases(flatten)

    cum_len=[0]+np.cumsum(lengths).tolist()
    tags=[flatten_tags[cum_len[i]:cum_len[i+1]] for i in range(len(cum_len)-1)]

    return tags, marks

def rewrite_func(data):
    sentence, tags, mark=data
    phrases, new_tags = marks_to_phrases(mark, tags, sentence)
    new_sentence = rewrite(phrases_bag, sentence, tags, phrases, lm, strategy='beamsearch')
    # print(new_sentence)
    return new_sentence


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.device

    data_dir=FLAGS.data_dir
    with open(os.path.join(data_dir,FLAGS.input_file), 'r', encoding='utf8') as f:
        inputs = f.read().splitlines()
        if FLAGS.rewrite_num:
            inputs=inputs[:FLAGS.rewrite_num]
        # raws,_=list(zip(*[n.split(' | ',1) for n in inputs]))
        raws=[n.split(' | ',1)[0] for n in inputs]


    mode=FLAGS.mode
    mode_dict={'tag_to_mark':tag_to_mark,
               'tag_and_mark': tag_and_mark,
               'mark_to_tag': mark_to_tag}

    pred_tags,marks=mode_dict[mode](raws)

    with open(os.path.join('/nfs/users/liuchang/comments_dayu/tag_prediction/data','phrases_bag_no_brands.json'),
                    'r', encoding='utf8') as f:
        phrases_bag = json.load(f)

    # rewrites=[]l'c
    lm=CalScore('/nfs/users/liuchang/bert_extract_phrase/unigram_probs_model.json')

    # for sentence,tags,mark in zip(raws, pred_tags,marks):
    #     phrases,new_tags=marks_to_phrases(mark,tags,sentence)
    #     new_sentence=rewrite(phrases_bag,sentence,tags,phrases,lm,strategy='beamsearch')
    #     # print(new_sentence)
    #     rewrites.append(new_sentence)

    # pool=Pool(10)
    #
    # rewrites=pool.map(rewrite_func,zip(raws,pred_tags,marks),chunksize=100)

    rewrites=[rewrite_func(data) for data in zip(raws,pred_tags,marks)]

    timestr = time.strftime("%y-%m-%d_%H-%M-%S")
    with open(os.path.join('/nfs/users/liuchang/comments_dayu/tag_prediction/rewrite','{}/rewrites_'.format(mode) + timestr), 'w', encoding='utf8') as f:
        f.writelines([n+'\n' for n in rewrites])

    with open(os.path.join('/nfs/users/liuchang/comments_dayu/tag_prediction/rewrite','{}/compare_'.format(mode) + timestr), 'w', encoding='utf8') as f:
        for rewrite,tags,input in zip(rewrites,pred_tags,inputs):
            f.write(input+'\n'+rewrite+' | '+' # '.join(tags)+'\n\n')

