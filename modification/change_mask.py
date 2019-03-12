import re
import os

def first_mask(corpus,labels,out_path):
    corpus=[s+'\n' if not s.endswith('\n') else s for s in corpus  ]
    labels=[s+'\n' if not s.endswith('\n') else s for s in labels  ]
    results=[]
    for c ,l in zip(corpus,labels):
        s=c.replace('<mask>',l)
        s=re.sub(r'\[sep\][^，]+，',r'[sep] <mask> ，',s)
        results.append(s)
    with open(out_path, 'w', encoding='utf8') as f:
        f.writelines(results)

wdir='/data/share/liuchang/car_comment/mask/'
p1=os.path.join(wdir,'')
with open(p1, 'r', encoding='utf8') as f:
    corpus=f.readlines()
p2='/data/share/liuchang/car_comment/mask/'
with open(p2, 'r', encoding='utf8') as f:
    labels=f.readlines()

first_mask(corpus,labels,)