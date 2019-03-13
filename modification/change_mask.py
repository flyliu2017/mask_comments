import re
import os

def first_mask(out_dir,suffix):
    p1 = os.path.join(out_dir, '{}_corpus_{}'.format('test', suffix))
    with open(p1, 'r', encoding='utf8') as f:
        corpus = f.readlines()
    p2 = os.path.join(out_dir, '{}_labels_{}'.format('test', suffix))
    with open(p2, 'r', encoding='utf8') as f:
        labels = f.readlines()

    corpus=[s+'\n' if not s.endswith('\n') else s for s in corpus  ]
    labels=[s+'\n' if not s.endswith('\n') else s for s in labels  ]
    cs=[]
    ls=[]
    for c ,l in zip(corpus,labels):
        s=c.replace('<mask>',l)
        p=re.compile(r'\[sep\][^，]+，')
        newl=p.findall(s)[0]
        s=p.sub(r'[sep] <mask> ，',s)
        # s=re.sub(r'\[sep\][^，]+，',r'[sep] <mask> ，',s)
        cs.append(s)
        ls.append(newl)

    with open(os.path.join(out_dir,'first_mask_corpus_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.writelines(cs)
    with open(os.path.join(out_dir,'first_mask_labels_{}'.format(suffix)), 'w', encoding='utf8') as f:
        f.writelines(ls)


out_dir='/data/share/liuchang/car_comment/mask/p5_p10'
