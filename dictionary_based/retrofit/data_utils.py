import numpy as np
import time
import config
from collections import defaultdict

GOOD_LINES=set([])

def get_words(data_path, seeds):
    seeds=set(seeds)
    all_words=set()
    count=defaultdict(int)
    for line in open(data_path, 'r'):
        tmp=[]
        try:
            line, weight = line.strip().spilt('\t')
        except:
            pass
        parts=line.strip().split('/')
        for part in parts:
            words=part.split(' ')
            if len(words)>config.sen_len:continue
            tmp=tmp+words
        global GOOD_LINES
        for word in tmp:
            count[word]+=1
            if word in seeds or count[word]>=5:
                all_words.update(tmp)
                break
    return all_words

class SynonymDataset():
    def __init__(self, data_path, _word2id, batch_size=100, max_len=10,pre_compute=True):
        self.data_path=data_path
        self._word2id=_word2id
        self.batch_size=batch_size
        self.finished=False
        self.max_len=config.sen_len
        self.full=None
        start=time.time()
        if pre_compute:
            self.full=list(self.epoch_data(1))
            print('pre compute time', time.time()-start)
            print('one batches of steps', len(self.full))

    
    def padding(self, input):
        s=np.array([0,]*self.max_len, dtype=np.int32)
        l=min(len(input), self.max_len)
        s[:l]=np.array(input[:l],dtype=np.int32)
        return s

    def processing_line(self, line):
        try:
            line_,weight=line.strip().split('\t')
            weight=float(weight)
            line=line_
        except:
            #print 'error'
            weight=1
        parts=line.strip().split('/')
        parts=[[self._word2id[word] for word in part.split(' ') if word in self._word2id]\
              for part in parts if len(part.split(' '))<=config.sen_len]
        parts=[part for part in parts if part]
        lengths=[len(part) for part in parts]
        inputs=[self.padding(part) for part in parts]
        samples=[]
        for i in range(len(lengths)):
            for j in range(len(lengths)):
                if i==j:continue
                samples.append([inputs[i], inputs[j], lengths[i], lengths[j],weight])
                
                #print([inputs[i], inputs[j], lengths[i], lengths[j]])
                #raw_input('xxxxx')
        return samples
    
    #@threadsafe_generator 
    #@profile
    def epoch_data(self, epoch=10):
        if self.full:
            for i in range(epoch):
                for batch in self.full:
                    yield batch
            return
        tmp=[[],]*5
        #batch_data=
        processed=0
        for k in range(epoch):
            ii=open(self.data_path, 'r')
            s=0
            time1,time2=0,0
            start=time.time()
            for line in ii.readlines():
                s+=1
                if s%1000000==0:
                    print(s,time.time()-start,time1,time2)
                sstart=time.time()
                samples=self.processing_line(line)
                time1+=(time.time()-sstart)
                for sample in samples:
                    for j,t in enumerate(sample):
                        #print(j,t)
                        tmp[j]=tmp[j]+[t]
                #if (len(tmp[0])+processed)%200==0:
                #    print("time per step data get {}, steps={}".format((time.time()-start)/(len(tmp[0])+0.01+processed), len(tmp[0])+processed))
                sstart=time.time()
                if len(tmp[0])>=1*self.batch_size:
                    while len(tmp[0])>=self.batch_size:
                        batch=[tmp[j][:self.batch_size] for j in range(len(tmp))]
                        batch[:-1]=[np.array(ele, dtype=np.int32) for ele in batch[:-1]]
                        batch[-1]=np.array(ele, dtype=np.float32)
                        yield batch
                        processed+=self.batch_size
                        tmp=[tmp[j][self.batch_size:] for j in range(len(tmp))]
                time2+=(time.time()-sstart)
        while len(tmp[0])>0:
            l=min(self.batch_size,len(tmp[0]))
            batch=[tmp[j][:l] for j in range(len(tmp))]
            batch[:-1]=[np.array(ele, dtype=np.int32) for ele in batch[:-1]]
            batch[-1]=np.array(batch[-1], dtype=np.float32)
            yield batch
            tmp=[tmp[j][l:] for j in range(len(tmp))]
        self.finished=True
