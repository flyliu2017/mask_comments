from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import os
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler
#
# svm.fit(fs,scores)
# train_scores = svm.predict(fs)
# print(max(train_scores))
#
# mse = np.sum([(x - y) ** 2 for x, y in zip(train_scores, scores)])/len(scores)
#
# print(mse)




def compare_params(param_grid,fs,scores,test_fs,true_test_scores):
    mses=[]
    maxv=[]
    for c in param_grid['C']:
        l1=[]
        l2=[]
        for g in param_grid['gamma']:
            print('*'*20+'\n')
            print('C:{} , gamma:{} \n'.format(c,g))

            svm = SVR(C=c, gamma=g)
            svm.fit(fs,scores)
            train_scores = svm.predict(fs)
            print(max(train_scores))

            mse = np.sum([(x - y) ** 2 for x, y in zip(train_scores, scores)])/len(scores)

            print(mse)

            test_scores = svm.predict(test_fs)
            print(max(test_scores))

            mse = np.sum([(x - y) ** 2 for x, y in zip(test_scores, true_test_scores)])/len(test_scores)

            print(mse)

            l1.append(mse)
            l2.append(max(test_scores))
        mses.append(l1)
        maxv.append(l2)

    return mses,maxv

def gridsearch(model, param_grid, features, scores):
    gs=GridSearchCV(model,param_grid=param_grid,cv=5,n_jobs=4,verbose=2,iid=True)

    gs.fit(features,scores)

    print(gs.best_score_)
    print(gs.best_params_)

def generate_high_scores(features, scores,threshold, replicas=1):
    # f_mean=np.sum(np.array(features),0)/len(features)
    # s_mean=np.sum(scores)/len(scores)
    zipped=[(f,s) for f,s in zip(features,scores) if s>=threshold]
    features,scores=list(zip(*zipped))
    f_std=np.std(features,0)
    # s_std=np.std(scores)
    features=np.array(features * replicas)
    scores=np.array(scores * replicas)
    rand_nums=np.random.normal(0,f_std/20,features.shape)

    rand_features=features+rand_nums
    # rand_scores = np.random.normal(s_mean, s_std, num)
    # if num>1:
    #     minf=np.min(features,0)
    #     maxf=np.max(features,0)
    #     scaled=[[]]*num
    #     # rand_features=list(zip(*rand_features))
    #
    #     for i in range(3):
    #         scaled=np.concatenate([scaled,MinMaxScaler((minf[i],maxf[i])).fit_transform(np.reshape(rand_features[:,i],(-1,1)))],1)
    #
    #     rand_scores=MinMaxScaler((min(scores),max(scores))).fit_transform(rand_scores.reshape([-1,1])).reshape([1,-1])[0]

    return rand_features,scores

def scale_features(min_value,max_value,features):
    scaler=MinMaxScaler((min_value,max_value))
    result=[[]]*len(features)
    for feature in zip(*features):
        result=np.concatenate([result,scaler.fit_transform(np.reshape(feature,(-1,1)))],1)
    return result

if __name__ == '__main__':
    data_dir = '/data/share/liuchang/car_comment/mask/p5_p10/keywords/only_mask'

    with open('/data/share/liuchang/car_comment/mask/mask_comments/data_process/features2.tsv', 'r',
              encoding='utf8') as f:
        fs = f.readlines()
    with open('/data/share/liuchang/car_comment/mask/mask_comments/data_process/test_features.tsv', 'r',
              encoding='utf8') as f:
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

    fs = [s.strip().split('\t') for s in fs]
    fs = [[float(s) for s in l] for l in fs]
    test_fs = [s.strip().split('\t') for s in test_fs]
    test_fs = [[float(s) for s in l] for l in test_fs]
    scores = [float(s) for s in scores]
    true_test_scores = [float(s) for s in true_test_scores]

    svm = SVR(C=100, gamma=10)

    param_grid = {
        'gamma': [0.1, 1,5, 8,10,12,15],
        'C': [1, 10, 100, 1000, 2000]
        # 'C': [100]
    }
    rand_samples=generate_high_scores(fs,scores,threshold=3,replicas=5)
    fs=fs+list(rand_samples[0])
    scores=scores+list(rand_samples[1])
    # gridsearch(svm,param_grid,fs,scores)
    compare_params(param_grid,fs,scores,test_fs,true_test_scores)
    # print(rand_samples)