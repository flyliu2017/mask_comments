import re
import os

def first_mask(out_dir,suffix):
    p1 = os.path.join(out_dir, '{}_corpus_{}'.format('test', suffix))
    with open(p1, 'r', encoding='utf8') as f:
        corpus = f.read().split('\n')
    p2 = os.path.join(out_dir, '{}_labels_{}'.format('test', suffix))
    with open(p2, 'r', encoding='utf8') as f:
        labels = f.read().split('\n')

    cs=[]
    ls=[]
    pattern = re.compile('_([^_，]+)，')
    for c, l in zip(corpus, labels):
        s = c.replace('<mask>', l)
        m = pattern.search(s)
        if not m:
            raise ValueError(s)
        newl = m.group(1).strip()
        s = pattern.sub('_ <mask> ，', s)
        # s=re.sub(r'\[sep\][^，]+，',r'[sep] <mask> ，',s)
        cs.append(s)
        ls.append(newl)

    with open(os.path.join(out_dir,'first_mask_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(cs))
    with open(os.path.join(out_dir,'first_mask_labels_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(ls))

def recover_mask(out_dir,suffix):
    p1 = os.path.join(out_dir, '{}_corpus_{}'.format('test', suffix))
    with open(p1, 'r', encoding='utf8') as f:
        corpus = f.read().split('\n')
    p2 = os.path.join(out_dir, '{}_labels_{}'.format('test', suffix))
    with open(p2, 'r', encoding='utf8') as f:
        labels = f.read().split('\n')

    result=[c.replace('<mask>',l).split('[sep]')[-1] for c,l in zip(corpus,labels)]

    with open(os.path.join(out_dir,'no_mask_test_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(result))
    
    raw=[re.split('__.+?__',s)[-1] for s in result]
    with open(os.path.join(out_dir,'raw_test_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(raw))
        
    return result,raw
    
    

def change_mask_and_keywords(out_dir,suffix,index=0):
    p1 = os.path.join(out_dir, '{}_corpus_{}'.format('test', suffix))
    with open(p1, 'r', encoding='utf8') as f:
        corpus = f.read().split('\n')
    p2 = os.path.join(out_dir, '{}_labels_{}'.format('test', suffix))
    with open(p2, 'r', encoding='utf8') as f:
        labels = f.read().split('\n')

    with open(os.path.join(out_dir, 'phrase_keywords_lists_{}'.format(suffix)), 'r', encoding='utf8') as f:
        keywords = f.read().split('\n')
        keywords_list=[k.split(' ||| ') for k in keywords]

    cs = []
    ls = []
    pattern = re.compile('_([^_，]+)，')
    for c, l ,k in zip(corpus, labels,keywords_list):
        s = c.replace('<mask>', l)
        m = pattern.search(s)
        if not m:
            raise ValueError(s)
        newl = m.group(1).strip()
        s = pattern.sub('_ <mask> ，', s)
        # s=re.sub(r'\[sep\][^，]+，',r'[sep] <mask> ，',s)
        s=re.sub('.*\[sep\] ',k[index],s)
        cs.append(s)
        ls.append(newl)

    with open(os.path.join(out_dir, 'first_mask_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(cs))
    with open(os.path.join(out_dir, 'first_mask_labels_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.write('\n'.join(ls))

if __name__ == '__main__':
    
    out_dir='/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask'
    suffix='only_mask'
    # first_mask(out_dir,suffix)
    # recover_mask(out_dir,suffix)
    change_mask_and_keywords(out_dir,suffix,0)