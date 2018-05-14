# encoding=utf-8
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from nlp import load_w2v,save_w2v
from utils.base import dict_reverse
from data_utils import get_words, SynonymDataset
import config

class Word2VecRetrofit():

    def __init__(self):
        print 'load original w2v...'
        w2v = load_w2v(config.originalw2v_path)
        id2word={id_:key for id_,key in enumerate(w2v.keys())}
        word2id = dict_reverse(id2word)
        vectors=w2v.values()
        old_len=len(word2id)
        
        print 'load original w2v finished'

        print 'load synonym words...'
        synonym_words = get_words(config.synonym_path, w2v.keys())
        print 'load synonym words finished'

        print 'update w2v ...'
        #synonym_words=list(synonym_words)[:50000]
        for word in set(synonym_words)-set(word2id.keys()):
            id_=len(id2word)
            id2word[id_] = word
            word2id[word] = id_
            #vectors.append(np.zeros((config.vec_len),dtype=np.float64))
        append_vectors = np.random.uniform(-0.1,0.1,(len(word2id)-old_len, config.vec_len))
        vectors=np.concatenate([np.array(vectors,dtype=np.float16),\
                np.array(append_vectors,dtype=np.float16)], axis=0)
        alpha=old_len*[[1],]+(len(word2id)-old_len)*[[0],]
        self.word2id=word2id
        self.id2word=id2word
        print 'old number of words  = ', old_len
        print 'new number of words  = ', len(word2id)

        print 'build graph...'
        with tf.device('/cpu:0'):
            self.build_graph(vectors, alpha)
        print 'build graph finished'
        
    def build_graph(self, vectors, alpha):
        self.W = W =tf.Variable(np.array(vectors), dtype=config.dtype)
        self.W_old = W_old=tf.constant(np.array(vectors), dtype=config.dtype)
        Alpha=tf.constant(np.array(alpha), dtype=config.dtype)
        def castf(x):
            return tf.cast(x,config.dtype2)

        def test(seq, length, W): 
            seq=tf.nn.embedding_lookup(W, seq)
            mask=tf.expand_dims(tf.sequence_mask(length, config.sen_len, dtype=config.dtype),-1)
            ave=tf.reduce_sum(castf(seq)*castf(mask), axis=1)
            return tf.reduce_max(ave)
        def average(seq, length, W): 
            seq=tf.nn.embedding_lookup(W, seq)
            mask=tf.expand_dims(tf.sequence_mask(length, config.sen_len, dtype=config.dtype),-1)
            ave=tf.reduce_sum(castf(seq)*castf(mask), axis=1)
            return ave

        input1 = tf.placeholder(tf.int32, [None,config.sen_len])
        input2 = tf.placeholder(tf.int32, [None,config.sen_len])
        input_length1 = tf.placeholder(tf.int32, [None])
        input_length2 = tf.placeholder(tf.int32, [None])
        self.lr = learning_rate = tf.placeholder(tf.float32, shape=[])
        self.inputs=[input1, input2, input_length1, input_length2, learning_rate]
        ave1=average(input1, input_length1, W)
        ave2=average(input2, input_length2, W) 
        t1=test(input1, input_length1, W)
        t2=test(input2, input_length2, W) 

        def cosin_distance(a,b):
            normalize_a = tf.nn.l2_normalize(a,1)
            normalize_b = tf.nn.l2_normalize(b,1)
            loss=1-tf.reduce_mean(tf.reduce_sum(tf.multiply(normalize_a,normalize_b),1),0)
            return loss
        self.synonym_loss = synonym_loss = cosin_distance(ave1, ave2)

        input_ = tf.concat([input1, input2], axis=0)
        length_ = tf.concat([input_length1, input_length2], axis=0)
        mask = tf.expand_dims(tf.sequence_mask(length_, config.sen_len, dtype=config.dtype2),-1)
        offset = (tf.nn.embedding_lookup(W, input_) -\
                tf.nn.embedding_lookup(W_old, input_))*\
                 tf.nn.embedding_lookup(Alpha, input_)
        self.offset_loss = offset_loss =\
            tf.reduce_sum(tf.square(castf(offset))*mask)/tf.cast(tf.reduce_sum(length_),config.dtype2)
         
        self.loss= synonym_loss + offset_loss
        #self.loss=offset_loss
        #self.loss=synonym_loss
        #create an optimizer
        
        #params = tf.trainable_variables()
        #opt = tf.train.GradientDescentOptimizer(config.learning_rate)
        #gradients = tf.gradients(self.loss, params)
        #clipped_gradients, norm = tf.clip_by_global_norm(gradients,config.max_gradient_norm)
        #self.train_op = opt.apply_gradients(zip(clipped_gradients, params))
       
        #self.train_op = opt.apply_gradients(zip(gradients, params))

        self.train_op=tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        #self.train_op=tf.train.AdamDescentOptimizer(0.005).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver=tf.train.Saver()
    
    def train(self):
        print 'STRTA TRAIN'
        self. synonym_dataset = SynonymDataset(config.synonym_path, self.word2id, batch_size = config.batch_size)
        #session_conf = tf.ConfigProto(
            #device_count = {'CPU': 20, 'GPU':0}, 
        #    allow_soft_placement=True,
        #    log_device_placement=False,)
        with tf.Session() as sess:
            sess.run(self.init)
            learning_rate = config.learning_rate
            self.run_eval(sess)
            for k,input_data in enumerate(self.synonym_dataset.epoch_data(config.epochs)):
                #for d in input_data:
                #    print d.shape
                fd=dict(zip(self.inputs, input_data+[learning_rate]))
                #W=sess.run(self.W, feed_dict=fd)
                #print 'ffff',W.max()
                #sess.run(self.outs, feed_dict=fd)        
                loss,_ = sess.run([self.loss,self.train_op], feed_dict=fd) 
                if k%config.eval_steps==0 and k!=0:
                    self.run_eval(sess)
                    self.saver.save(sess,config.save_path,k)
                    self.save_w2v(sess)
                    #if k>config.pre_steps:
                    #    learning_rate*=0.9

    def run_eval(self,sess):
        sum_loss=[0,0]
        for k,input_data in enumerate(self.synonym_dataset.epoch_data(1)):
            if k%7!=0:continue
            fd=dict(zip(self.inputs, input_data+[config.learning_rate]))
            loss = sess.run([self.synonym_loss, self.offset_loss], feed_dict=fd)
            sum_loss=[sum_loss[0]+loss[0], sum_loss[1]+loss[1]]
        print 'eval loss=', sum_loss
        
    def save_w2v(self,sess):
        w2v = OrderedDict()
        vectors=sess.run(self.W)
        for id_, word in self.id2word.iteritems():
            w2v[word]=vectors[id_]
        save_w2v(w2v,config.new_vec_path)
            

def main():
    model = Word2VecRetrofit()
    model.train()

if __name__=="__main__":
    main() 
