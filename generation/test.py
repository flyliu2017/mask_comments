import os
dir='/data/share/liuchang/car_comment/mask/context'
path1='corpus_2000_prefixed'
path2='labels_2000_prefixed'

with open(os.path.join(dir,path1), 'r', encoding='utf8') as f:
    corpus = f.read().split('\n')
with open(os.path.join(dir,path2), 'r', encoding='utf8') as f:
    labels = f.read().split('\n')

r=[]
for c,l in zip(corpus,labels):
    s=c.replace('<mask>',l)
    r.append(s)

with open(os.path.join(dir,'2000_prefixed'), 'w', encoding='utf8') as f:
    f.write('\n'.join(r))