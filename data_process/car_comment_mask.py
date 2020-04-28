#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import jieba
import pandas
import pickle
import os
import re


# In[2]:


data_dir='/nfs/users/liuchang/comments_dayu'


# In[2]:



# In[3]:

comments_str=pandas.read_json('/nfs/users/liuchang/car_comment/mask/comments_5_80_p5_p10.json',orient='index')


# In[4]:


car_types=pandas.read_json('/nfs/users/liuchang/car_comment/mask/car_types.json',orient='index',typ = 'series')


# In[5]:


car_types=pandas.DataFrame(car_types,columns=['seriesname'])


# In[6]:


comments_str=comments_str.join(car_types)


# In[49]:



# In[7]:


comments_str=comments_str.rename(columns={'为什么选择这款车':'为什么最终选择这款车','最不满意':'最不满意的一点',
                    '最满意':'最满意的一点'})

# In[14]:


label_list=list(comments_str.keys())
label_list.remove('seriesname')
label_list=label_list+['其它描述']


# In[15]:




# In[92]:



# In[114]:



# In[23]:


df_join=pandas.read_json(os.path.join(data_dir,'df_join.json')).sort_index()

# In[32]:


comments_joined=df_join.loc[:890286][label_list]


# In[37]:


from multiprocessing.pool import Pool


# In[38]:


pool=Pool(8)


# In[ ]:


def cut(string_list):
    l=[' '.join(jieba.cut(s)) for s in string_list]
    return l


# In[ ]:


size=(len(comments_joined)+7)//8
comments_list=pool.map(cut,comments_joined.values,chunksize=size)


# In[ ]:


comments_list=comments_list.applymap(lambda l:[w for w in l if w.strip() !='' ])


types=df['seriesname'].values
types=np.tile(types,12)
types=types.reshape((1,-1))[0]
classes=df.columns[:-1]
classes=list(classes)
classes=classes*len(df)
classes=['__{}__'.format(n) for n in classes]
types=['__{}__'.format(n) for n in types]
prefix=[t+' '+c+' ' for c,t in zip(classes,types)]

# In[ ]:


comments_str=comments_list.applymap(lambda s:' '.join(s))


# In[ ]:



# In[321]:


comments_str.to_json(dir+'comments_str.json',force_ascii=False,orient='index')


# In[54]:


m=lambda x: x if len(x)>=5 and len(x)<=80 else []
comments_list_5_80=comments_list.applymap(m)


# In[55]:


comments_str_5_80=comments_list_5_80.applymap(lambda s:' '.join(s))


# In[16]:


import re


# In[139]:


df.to_json(dir+"mask/df.json",force_ascii=False)


# In[194]:


def mask_subsentence_str(string, min_sub_num=2,mask_index=None,mode='random',min_index=0,max_index=-1):
    l=re.split('[，。]',re.sub('[!?;！？；]|… …|…','，',string))
    l=[s for s in l if s.strip()!='']
    if len(l)<min_sub_num:
        return ''
    
    if not mask_index:
#         print("No mask_index,check mode.")
        if mode == 'random':
                mask_index=np.random.randint(min_index, len(l)+max_index+1)
        elif mode == 'median':
            mask_index = [len(l) // 2 for l in lists]
        else:
            raise ValueError('mode should be "random" or "median"')
            
    labels = l[mask_index]
    corpus = '，'.join(l[:mask_index] + [' <mask> '] + l[mask_index + 1:])
    
    return corpus+'<separate>'+labels


# In[56]:


comments_str_5_80


# In[195]:


comments_masked_str_5_80=comments_str_5_80.applymap(lambda s: mask_subsentence_str(s) )


# In[196]:


comments_masked_str_5_80.to_json(dir+'mask/comments_masked_str_5_80.json')


# In[198]:


comments_masked_str_5_80['动力'][1]


# In[199]:


masked_classe_type_5_80=comments_masked_str_5_80.copy()


# In[200]:


types=df['question_forum'].map(lambda s: '__'+s+'__ ')


# In[221]:


types.describe()


# In[201]:


for l in label_list:
    masked_classe_type_5_80[l]=pandas.Series(['__'+l+'__ ']*len(masked_classe_type_5_80)).str.cat(masked_classe_type_5_80[l])
    masked_classe_type_5_80[l]=types.str.cat(masked_classe_type_5_80[l])


# In[202]:


masked_classe_type_5_80


# In[203]:


masked_classe_type_5_80['性价比'][0]


# In[204]:


l=masked_classe_type_5_80.values.flatten()


# In[205]:


l=[s.split('<separate>') for s in l]


# In[206]:


l=[n for n in l if len(n)==2]


# In[207]:


len(l)


# In[208]:


corpus,labels=zip(*l)


# In[210]:


labels[16]


# In[211]:


with open(dir+'mask/corpus_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(corpus))


# In[212]:


with open(dir+'mask/labels_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(labels))


# In[213]:


length=len(labels)


# In[214]:


with open(dir+'mask/train_corpus_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(corpus[:int(0.8*length)]))
with open(dir+'mask/eval_corpus_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(corpus[int(0.8*length):int(0.9*length)]))
with open(dir+'mask/test_corpus_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(corpus[int(0.9*length):]))


# In[215]:


with open(dir+'mask/train_labels_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(labels[:int(0.8*length)]))
with open(dir+'mask/eval_labels_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(labels[int(0.8*length):int(0.9*length)]))
with open(dir+'mask/test_labels_5_80.txt','w',encoding='utf8') as f:
    f.writelines('\n'.join(labels[int(0.9*length):]))


# In[222]:


def compare_result(preds,labels,corpus,output='compare.txt'):
    with open(preds,'r') as f:
        pl=f.read().splitlines()
        if pl[-1]=='':
            pl.pop()
    with open(labels,'r') as f:
        ll=f.read().splitlines()
        if ll[-1]=='':
            ll.pop()
    with open(corpus,'r') as f:
        cl=f.read().splitlines()
        if cl[-1]=='':
            cl.pop()
    if len(pl)!=len(ll) or len(ll)!=len(cl):
        raise ValueError("predictions,labels and corpus should have same length.")
    result=[]
    for p,l,c in zip(pl,ll,cl):
        result.append(c.replace('mask',l +" | "+p))
    with open(output,'w',encoding='utf8') as f:
        f.writelines('\n'.join(result))


# In[223]:


dir2=r"/nfs/users/liuchang/car_comment/mask/"


# In[224]:


preds=dir2+'pred_200000.txt'
labels=dir2+'test_labels.txt'
corpus=dir2+'test_corpus.txt'


# In[225]:


compare_result(preds,labels,corpus,dir2+'compare.txt')


# In[329]:


df.to_json(dir+'mask/df.json',force_ascii=False,orient='index')


# In[274]:


df=df.sort_index()


# In[300]:


df['question_forum'].to_json(dir+'mask/car_types.json',force_ascii=False)


# In[302]:


types=pandas.read_json(dir+'mask/car_types.json',typ='series')


# In[89]:


df2=pandas.read_json(dir+'mask/df.json',orient='index')


# In[97]:


s=df2[df2['question_forum']=='明锐']['操控']


# In[98]:


s.loc[250]


# In[99]:


df2


# In[12]:


lengths=[len(s.split(' ')) for s in l]


# In[34]:


l


# In[17]:


s=pandas.Series(lengths)


# In[18]:


s.describe()


# In[24]:


s.plot.area()


# In[31]:


comments=df2[label_list]


# In[86]:


comments


# In[41]:


cl=comments.values.flatten()


# In[49]:


cll=[len(s) for s in cl]


# In[50]:


clls=pandas.Series(cll)


# In[51]:


clls.describe()


# In[5]:


import re


# In[12]:


l=[re.sub('_[^_，]+，','_ <mask> ，',s) for s in l]


# In[101]:


comments_str=pandas.read_json(dir+'mask/df_5_80_p5_p10.json',orient='index')


# In[102]:


comments_str


# In[47]:


l=comments_str.values.flatten()


# In[13]:


from nltk.probability import FreqDist


# In[52]:


fd=FreqDist(l)


# In[84]:


d=dict(fd)


# In[59]:


for k in d.keys():
    d[k]=d[k]/n


# In[57]:


n=fd.N()


# In[61]:


import json


# In[85]:


with open(dir+'mask/unigram_model.json','w',encoding='utf8') as f:
    json.dump(d,f)


# In[27]:


import re


# In[48]:


l=[re.sub('[!?;！？；。]|… …|…','，',string) for string in l]


# In[49]:


l=[s.split(' ') for s in l]


# In[39]:


l=np.array(l).flatten()


# In[50]:


l=[s for ls in l for s in ls if s != '' ]


# In[81]:


fd['<unk>']=1


# In[82]:


b=fd.B()
for k in fd.keys():
    fd[k]=(fd[k]+1)/(n+b)


# In[83]:


fd


# In[79]:


def generate_unigram_model(corpus,vocab):
    fd=FreqDist(corpus)
    n=fd.N()
    keys=fd.keys()
    d2={}
    for k in keys:
        if k in vocab:
            d2[k]=fd[k]
    d2['<unk>']=n-sum(d2.values())
    for k in d2.keys():
        d2[k]=d2[k]/n
    with open(dir+'mask/unigram_model.json', 'w', encoding='utf8') as f:
        json.dump(d2,f)


# In[67]:


with open(dir+'mask/p5_p10/vocab_p5_p10', 'r', encoding='utf8') as f:
    vocab=f.read().splitlines()


# In[80]:


generate_unigram_model(l,vocab)


# In[4]:


with open('/nfs/users/liuchang/ch-en/train.ch','r',encoding='utf8') as f:
    zh=f.readlines()


# In[7]:


import random
index=random.sample(range(int(1e7)),int(1e6))


# In[10]:


train_small=np.array(zh)[index]


# In[11]:


len(zh)


# In[12]:


train_small


# In[13]:


with open('/nfs/users/liuchang/ch-en/train.en','r',encoding='utf8') as f:
    en=f.readlines()


# In[14]:


en_small=np.array(en)[index]


# In[15]:


with open('/nfs/users/liuchang/ch-en/train_small.ch','w',encoding='utf8') as f:
    f.writelines(train_small)


# In[16]:


with open('/nfs/users/liuchang/ch-en/train_small.en','w',encoding='utf8') as f:
    f.writelines(en_small)


# In[3]:


import json
with open('/nfs/users/liuchang/car_comment/mask/phrase_prob_pruned_1e7.json', 'r', encoding='utf8') as f:
     pd=json.load(f)
        


# In[4]:


v=list(pd.values())


# In[8]:


k=list(pd.keys())


# In[16]:


n=sum(np.exp(v))+0.5


# In[ ]:


with open('/nfs/users/liuchang/car_comment/mask/phrase_log_prob.pickle', 'rb') as f:
    phrase_log_prob = pickle.load(f)


# In[22]:


vs=list(phrase_log_prob.values())


# In[26]:


n=len(phrase_log_prob)+0.5


# In[28]:


pd={key:value for key,value in zip(k,v)}


# In[27]:


v=[s-np.log(0.5/n) for s in v]


# In[29]:


len(pd)


# In[30]:


pd['<oov>']=np.log(0.5/n)


# In[31]:


pd['<oov>']


# In[32]:


with open('/nfs/users/liuchang/car_comment/mask/phrase_prob_pruned_1e7.json', 'w', encoding='utf8') as f:
     json.dump(pd,f)


# In[33]:


del phrase_log_prob


# In[34]:


with open('/nfs/users/liuchang/car_comment/mask/zh2en_prob.pickle', 'rb') as f:
    z2e_prob_dist = pickle.load(f)


# In[35]:


import time


# In[36]:


pruned={}
i=0
for zh in pd:
    if i%1000==0:
        print(i)
    en_prob=z2e_prob_dist[tuple(zh.split(' '))]
    pruned[zh]={' '.join(k):en_prob[k] for k in en_prob}
    i+=1


# In[43]:


with open('/nfs/users/liuchang/car_comment/mask/z2e_probdist_pruned_1e7_2.json', 'w',encoding='utf8') as f:
    json.dump(pruned,f)


# In[38]:


len(pruned)


# In[39]:


ks=set(k)-set(pruned.keys())


# In[41]:


i=0
for zh in ks:
    if i%1000==0:
        print(i)
    en_prob=z2e_prob_dist[tuple(zh.split(' '))]
    pruned[zh]={' '.join(k):en_prob[k] for k in en_prob}
    i+=1


# In[3]:


from nltk.probability import FreqDist


# In[46]:


i=0
for zh in pruned:
    if i%1000==0:
        print(i)
    pd2 = pruned[zh]
    fd = FreqDist()
    pruned[zh] = fd
    for en in pd2:
        if pd2[en] >= 0.01:
            fd[en] = pd2[en]
    n = fd.N()
    for en in fd:
        fd[en] = fd[en] / n
    i+=1


# In[4]:


with open('/nfs/users/liuchang/car_comment/mask/z2e_probdist_pruned_1e7_0.01.json', 'r',encoding='utf8') as f:
    pruned=json.load(f)


# In[4]:


os.chdir('/nfs/users/liuchang/car_comment/mask/mask_comments')


# In[5]:


with open('/nfs/users/liuchang/car_comment/mask/phrase_prob_pruned_1e7.json', 'r' ,encoding='utf8') as f:
     pd=json.load(f)


# In[6]:


pd=FreqDist(pd)


# In[7]:


oov=pd['<oov>']


# In[8]:


n=0.5/np.exp(oov)


# In[9]:


n


# In[10]:


for zh in pd:
    pd[zh]=pd[zh]+oov+np.log((n-0.5)/n)


# In[11]:


topn=pd.most_common(1000000)


# In[13]:


topn[-1]


# In[16]:


pd2={}
for zh in topn:
    pd2[zh[0]]=pd[zh[0]]
pd2['<oov>']=oov


# In[21]:


with open('/nfs/users/liuchang/car_comment/mask/phrase_prob_pruned_1e6.json', 'w' ,encoding='utf8') as f:
     json.dump(pd2,f)


# In[20]:


pd2[topn[3500004][0]]


# In[22]:


pruned={zh[0]:pruned[zh[0]] for zh in topn}


# In[23]:


len(pruned)


# In[25]:


pruned[topn[9099][0]]


# In[26]:


with open('/nfs/users/liuchang/car_comment/mask/z2e_probdist_pruned_1e6_0.01.json', 'w',encoding='utf8') as f:
    json.dump(pruned,f)


# In[27]:


topn[9099][0]


# In[1]:


from sklearn.svm import SVR
import os
import numpy as np

data_dir = '/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask'

with open('/nfs/users/liuchang/car_comment/mask/mask_comments/data_process/features2.tsv', 'r', encoding='utf8') as f:
    fs = f.readlines()
with open('/nfs/users/liuchang/car_comment/mask/mask_comments/data_process/test_features.tsv', 'r', encoding='utf8') as f:
    test_fs = f.readlines()

with open(os.path.join(data_dir, 'train_rewrite_pairs_1547.tsv'), 'r',
          encoding='utf8') as f:
    paraphrase_scores = f.read().splitlines()

paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]

sentences, paraphrases, scores = list(zip(*paraphrase_scores))

with open(os.path.join(data_dir, 'test_rewrite_pairs2.tsv'), 'r',
          encoding='utf8') as f:
    paraphrase_scores = f.read().splitlines()

paraphrase_scores = [s.strip().split('\t') for s in paraphrase_scores]

ts, tp, true_test_scores = list(zip(*paraphrase_scores))

fs=[s.strip().split('\t') for s in fs]
fs=[[float(s) for s in l] for l in fs]
test_fs=[s.strip().split('\t') for s in test_fs]
test_fs=[[float(s) for s in l] for l in test_fs]
scores=[float(s) for s in scores]
true_test_scores=[float(s) for s in true_test_scores]


# In[96]:


svm=SVR(C=100,gamma=10)

svm.fit(fs,scores)
train_scores = svm.predict(fs)
print(max(train_scores))

mse = np.sum([(x - y) ** 2 for x, y in zip(train_scores, scores)])/len(scores)

print(mse)


# In[97]:


test_scores = svm.predict(test_fs)
print(max(test_scores))
mse = np.sum([(x - y) ** 2 for x, y in zip(test_scores, true_test_scores)])/len(test_scores)
print(mse)


# In[3]:


import matplotlib.pyplot as pl


# In[98]:


sl=[[test_scores[i]  for i in range(len(test_scores)) if true_test_scores[i]==n] for n in np.arange(1.0,5.2,0.25) ]
pl.boxplot(sl,labels=np.arange(1.0,5.2,0.25))


# In[2]:


os.chdir('/nfs/users/liuchang/car_comment/mask/mask_comments')

from data_process.pem import *


# In[3]:


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
scorer = CalScore('/nfs/users/liuchang/car_comment/mask/mask_comments/data_process/unigram_probs_model.json')

data_dir = '/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask'

with open('/nfs/users/liuchang/car_comment/mask/z2e_probdist_pruned_1e6_0.01.json', 'r') as f:
    z2e_prob_dist = json.load(f)
print('z2e_prob_dist loaded.')

with open('/nfs/users/liuchang/car_comment/mask/phrase_prob_pruned_1e6.json', 'r') as f:
    phrase_log_prob = json.load(f)
print('phrase_log_prob loaded.')

oov_value = phrase_log_prob['<oov>']

def oov():
    return oov_value

phrase_log_prob = defaultdict(oov, phrase_log_prob)
    
pem = PEM(z2e_prob_dist, phrase_log_prob, scorer)


# In[4]:


features=pem.pairs_to_features(sentences,paraphrases,8)


# In[5]:


test_fs=pem.pairs_to_features(ts,tp,8)


# In[40]:


ts=[]
for n in test_fs:
    ts.extend(list(zip(n[0],n[2])))


# In[42]:


fluency=pem.scorer.get_ppl_from_lm(tp)

fluency=[-np.log(n) for n in fluency]


# In[35]:


features=[(n[0],f,n[1]) for n,f in zip(fs,fluency) ]


# In[43]:


test_fs=[(n[0],f,n[1]) for n,f in zip(ts,fluency) ]


# In[37]:


features=[[str(n) for n in l] for l in features]
features=['\t'.join(n) for n in features]


# In[44]:


test_fs=[[str(n) for n in l] for l in test_fs]
test_fs=['\t'.join(n) for n in test_fs]


# In[45]:


test_fs


# In[39]:


with open('/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask/train_rewrite_pairs_1547.tsv'.format(type), 'w',
          encoding='utf8') as f:
    f.write('\n'.join(features))


# In[46]:


with open('/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask/test_features.tsv'.format(type), 'w',
          encoding='utf8') as f:
    f.write('\n'.join(test_fs))


# In[52]:


with open('/nfs/users/liuchang/car_comment/mask/p5_p10/keywords/only_mask/train_rewrite_pairs_1547.tsv'.format(type), 'w',
          encoding='utf8') as f:
    for s,p,score in zip(sentences,paraphrases,scores):
        f.write(s+'\t'+p+'\t'+str(score)+'\n')


# In[7]:


sen,para,hs=list(zip(*txts))


# In[8]:


pool=Pool(8)


# In[12]:


features = pool.apply_async(pem.pair_to_features,((sen[0],para[0]),))


# In[6]:


phrase_log_prob


# In[ ]:


def pair_to_features(self, rewrite_pair):
    sentence, paraphrase=rewrite_pair

    bpng = (sentence_bpng(sentence, self.phrase_log_prob, self.z2e_prob_dist),
             sentence_bpng(paraphrase, self.phrase_log_prob, self.z2e_prob_dist))
    adequacy = ngrams_f1(bpng[0], bpng[1])

    zh_ngram = (sentence_ngrams(sentence), sentence_ngrams(paraphrase))
    dissimilarity = ngrams_f1(zh_ngram[0], zh_ngram[1])

    ppl = self.scorer.get_ppl_from_lm(paraphrase)

    fluency = -np.log(ppl)

    return adequacy, fluency, dissimilarity


# In[75]:


feas=[[str(n) for n in l] for l in feas]


# In[76]:


feas=['\t'.join(n) for n in feas]


# In[4]:


with open('/nfs/users/liuchang/car_comment/mask/mask_comments/data_process/features3.tsv', 'r', encoding='utf8') as f:
    fs=f.readlines()


# In[5]:


len(fs)


# In[ ]:




