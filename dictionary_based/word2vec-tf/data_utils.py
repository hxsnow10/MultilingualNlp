# encoding=utf-8
import numpy as np

from tf_utils.data import Vocab
from tf_utils.data import LineBasedDataset

class word2vec_line_processing():

    def __init__(self, d_id2words, window, skip=True):
        self.vocab = Vocab(words=d_id2words)
        self.unk_id = self.vocab[self.vocab.unk]
        self.window = window
        self.skip=skip
        self.size=2
    
    def __call__(self, line):
        words=line.strip().split()
        ids=[ self.vocab.get(word, self.unk_id) for word in words]
        if self.skip:
            ids = [id_ for id_ in ids if id_!=self.unk_id]
        examples, labels=[], []
        for c in range(len(ids)):
            for k in range(max(-self.window, -c), min(self.window+1, len(ids)-c)):
                examples.append(ids[c])
                labels.append(ids[c+k])
        # examples = np.array(examples, dtype=np.int64)
        # labels = np.array(labels, dtype=np.int64)
        rval=[examples, labels]
        return rval

if __name__=="__main__": 
    dict_path='vocab.txt'
    d_id2words={k:line.strip().split()[0] for k,line in enumerate(open(dict_path, 'r'))}
    window=5
    p = word2vec_line_processing(d_id2words, window)
    dataset = LineBasedDataset(['zh.txt'],line_processing=p,len=2)
    k=0
    for k,s in enumerate(dataset):
        print k
        for ele in s:
            print '\t',ele.shape
        k+=1
        if k>=4:break
