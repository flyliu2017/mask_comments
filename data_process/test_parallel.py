from data_process.keywords import *
from multiprocessing.pool import Pool




def mask(s):
    return kp.mask_unimportant_words(s, ratio=ratio)

if __name__ == '__main__':

    data_dir = '/nfs/users/liuchang/car_comment/mask/p5_p10/keywords'

    corpus = read_to_list('/nfs/users/liuchang/car_comment/mask/selected_str_5_80_p5_p10.txt')

    STOP_WORDS = read_to_list('/nfs/users/liuchang/car_comment/mask/stop_words')
    corpus_keywords = read_to_dict('/nfs/users/liuchang/car_comment/mask/corpus_keywords', '\t', float, 1000)
    word_tfidf = read_to_dict('/nfs/users/liuchang/car_comment/mask/word_tfidf', '\t', float, None)
    num_words = read_to_list('/nfs/users/liuchang/car_comment/mask/mask_comments/data_process/key_words_all.txt')

    vectorizer = TfidfVectorizer(token_pattern=r'(?:^|(?<=\s))([^\s]+)(?=\s|$)', stop_words=STOP_WORDS,
                                 max_features=50000)
    vectorizer.fit(corpus)
    kp = Keywords_Processor(vectorizer,
                            stop_words=None,
                            word_tfidf=word_tfidf,
                            corpus_keywords=corpus_keywords,
                            num_words=num_words)

    ratio = 0.7
    suffix = 'only_mask'

    process_num=8
    pool = Pool(process_num)
    times = []

    for type in [ 'train','eval','test']:
        with open(os.path.join(data_dir, 'only_mask/no_mask_{}_corpus_{}'.format(type, suffix)), 'r', encoding='utf8') as f:
            raw = f.read().splitlines()

        prefix = [re.search(r'__.+?__ __.+?__', s).group() for s in raw]
        no_prefix = [raw[i].split(prefix[i])[-1].strip() for i in range(len(raw))]

        time1=time.time()
        chunksize=(len(no_prefix)+process_num-1)//process_num
        masked = pool.map(mask,no_prefix,chunksize=chunksize)
        time2=time.time()
        # print('cost time:{}'.format(time2-time1))
        times.append(time2-time1)

        masked = [p + m for p, m in zip(prefix, masked)]

        with open(os.path.join(data_dir, '{}_corpus_masked_{}'.format(type, ratio)), 'w', encoding='utf8') as f:
            f.write('\n'.join(masked))
        masked = [re.sub(r'(<mask> )+<mask>', '<mask>', s) for s in masked]
        with open(os.path.join(data_dir, '{}_corpus_infilling_masked_{}_keep_comma'.format(type, ratio)), 'w',
                  encoding='utf8') as f:
            f.write('\n'.join(masked))
        masked = [re.sub(r'<mask>( <mask>| ，)+', '<mask>', s) for s in masked]
        with open(os.path.join(data_dir, '{}_corpus_infilling_masked_{}'.format(type, ratio)), 'w', encoding='utf8') as f:
            f.write('\n'.join(masked))

    print(times)