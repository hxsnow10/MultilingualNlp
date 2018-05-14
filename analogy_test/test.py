#!/usr/bin/env python
#encoding=utf-8
from nlp import load_w2v, save_w2v
from scipy.stats.stats import pearsonr,spearmanr
from scipy.spatial.distance import cosine
import os   
import numpy as np
import argparse

def sim(vectors, words1, words2):
    for x in vectors:
        L=vectors[x].shape[0]
        break
    def sumvec(words):
        vec=np.zeros([L],dtype=np.float32)
        for word in words:
            if word in vectors:
                vec+=vectors[word]
            else:
                print word, 'not in vectors'
        return vec
    vec1 = sumvec(words1)
    vec2 = sumvec(words2)
    return 1-cosine(vec1, vec2)

def read_test(test_path):
    for line in open(test_path,'r'):
        try:
            words1,words2,score=line.strip().lower().split('\t')
            words1 = words1.split()
            words2 = words2.split()
            score = float(score)
            yield words1, words2, score
        except:
            print line,'error'

def measure_score(w2v, test_path):
    print test_path
    data=read_test(test_path)
    true=[]
    pred=[]
    for words1,words2,true_score in data:
        pred_score=sim(w2v, words1, words2)
        #raw_input('xxxxxxxxx')
        if str(pred_score)=='nan':continue
        print words1, words2, true_score, pred_score
        pred.append(pred_score)
        true.append(true_score)
        print pred_score
    true=np.array(true)
    pred=np.array(pred)
    print pearsonr(true, pred)[0]
    print spearmanr(true, pred)[0]
    #raw_input('xxxxxxxxxxx')
    score = (pearsonr(true, pred)[0] + spearmanr(true, pred)[0])/2
    return score    

def filter(vec_path, all_words, output_vec_path):
    w2v=load_w2v(vec_path)
    new_w2v={}
    for w in all_words:
        if w not in w2v:
            print w, 'not in w2v'
            continue
        v = w2v[w]
        new_w2v[w]=v
    save_w2v(new_w2v, output_vec_path)

def all_words(path):
    if not os.path.exists(path):return set([])
    elif os.path.isfile(path):
        ii=open(path,'r')
        rval=set([])
        for line in ii:
            for word in line.strip().split():
                #words=tok(word)
                rval.add(word)
        print path, len(rval)
        return rval
    elif os.path.isdir(path):
        rval=set([])
        for root, dirs, files in os.walk(path):
            for file_ in files:
                s=os.path.join(root, file_)
                rval.update(all_words(s))
        return rval

test_path='semeval17-2'
original_vec_path='new_vec.txt'
filter_vec_path='filtered_vec.txt'

def main_prepare():
    aw=all_words(test_path)
    print len(aw)
    #raw_input('xxxxxxxx')
    filter(original_vec_path, aw, filter_vec_path)

def test(vec_path):
    semeval17_main_test(vec_path)
import os
cur_dir = os.path.dirname( os.path.abspath(__file__)) or os.getcwd()
df_test_path=os.path.join(cur_dir,'semeval17-2')

def semeval17_main_test(vec_path, test_path=None):
    test_path = test_path or df_test_path
    
    w2v=load_w2v(vec_path)
    result={}
    print 'here'
    print test_path
    for root, dirs, files in os.walk(test_path):
        for file_ in files:
            s=os.path.join(root, file_)
            print s
            result[file_]=measure_score(w2v, s)
    for file_ in result:
        print file_, result[file_]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vec_path')
    args = parser.parse_args()
    semeval17_main_test(args.vec_path)
