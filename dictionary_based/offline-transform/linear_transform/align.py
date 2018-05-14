#!/usr/bin/env python
# coding: utf-8

# # So, show me how to align two vector spaces for myself!

# No problem. We're going to run through the example given in the README again, and show you how to learn your own transformation to align the French vector space to the Russian vector space.
# 
# First, let's define a few simple functions...

# In[1]:

import numpy as np
from nlp import load_w2v, save_w2v, load_synonym
from scipy.spatial.distance import cosine
import argparse
import sys
from MlingualEmd.analogy_test import test

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=False):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

def build_source_matrix(w2v1, w2v2, synonym, max_len=1, source='en', target='de'):
    def sum_vec(w2v, words):
        if len(words)>max_len or not words:return None
        v=np.array([0,]*w2v.dim, dtype=np.float32)
        for word in words:
            if word in w2v:
                v+=w2v[word]
            else:
                return None
        return v
    source_m, target_m, biwords=[],[],[]
    a={'en':0,'zh':0}
    for part1 in synonym:
        words1= part1.split('@')[-1].split()
        sl=part1.split('@')[0]
        if sl!=source:continue
        vec1=sum_vec(w2v1, words1)
        if vec1 is None: continue
        for part2 in synonym[part1]:
            words2=part2.split('@')[-1].split()
            tl=part2.split('@')[0]
            if tl!=target:continue
            vec2=sum_vec(w2v2, words2)
            if vec2 is None: continue
            source_m.append(vec1)
            target_m.append(vec2)
            biwords.append((words1[0],words2[0]))
    #for word in set(w2v1.keys()) & set(w2v2.keys()):
    #    source.append(w2v1[word])
    #    target.append(w2v2[word])
    #    biwords.append((word,word))

    source_m=np.array(source_m, dtype=np.float32)
    target_m=np.array(target_m, dtype=np.float32)
    return source_m, target_m, biwords

def cosine_s(v1,v2):
    return (1-cosine(v1,v2))/2+0.5

def test_similarity(vec1, vec2, test_samples):
    print('-'*40,'test samples started','-'*40)
    n,s=0,0
    for w1,w2 in test_samples:
        if w1 not in vec1 or w2 not in vec2:pass
        v1,v2=vec1[w1],vec2[w2]
        print w1,w2,cosine_s(v1,v2)
        n,s=n+1,s+cosine_s(v1,v2)
    print('SUMMARY:',n,s/n)
    print('-'*40,'test samples finished','-'*40)

def apply_transform(w2v,transmat):
    embed = np.array(w2v.values())
    embed = np.matmul(embed, transmat)
    for k,w in enumerate(w2v):
        w2v[w]=embed[k]
    return w2v
    
def align_word_vectors(w2v_path1, w2v_path2, dict_map_path, output_path,
        source_language='zh', target_language='en'):
    '''use dict_map to align w2v1 to space of w2v2
    '''
    languages = [source_language, target_language]
    print languages
    synonym = load_synonym( dict_map_path, languages=languages)
    # quit()
    vec1 = load_w2v( w2v_path1, np.float32)
    vec2 = load_w2v( w2v_path2, np.float32)
    source_matrix, target_matrix, biwords =\
        build_source_matrix(vec1, vec2, synonym, 1, source_language, target_language)
    print source_matrix.shape, target_matrix.shape
    test_similarity(vec1, vec2, biwords)
    transform = learn_transformation(source_matrix, target_matrix)
    vec1 = apply_transform(vec1, transform)
    test_similarity(vec1, vec2, biwords)

    save_w2v(vec1, output_path)
    for w in vec1:
        if w not in vec2:
            vec2[w]=vec1[w]
    output_path2=utput_path[:-4]+'merged.txt'
    save_w2v(vec2, output_path2)
    test(output_path2)

if __name__=="__main__":
    #align_word_vectors( 'vec1.txt', 'vec2.txt', 'synonym.tok', 'vec1_.txt',[('.','.')])
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--source_path')
    parser.add_argument('-t','--target_path')
    parser.add_argument('-m','--synonym_path', default='')
    parser.add_argument('-o','--output_vec_path', default='merged.txt')
    parser.add_argument('-sl','--source_language')
    parser.add_argument('-tl','--target_language')
    parser.add_argument('-log','--logs_path')
    args = parser.parse_args()
    import sys
    sys.stdout=open(args.logs_path,'w')
    align_word_vectors( args.source_path, args.target_path,  args.synonym_path, args.output_vec_path,
            args.source_language, args.target_language) 
