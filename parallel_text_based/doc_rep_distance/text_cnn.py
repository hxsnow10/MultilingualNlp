# encoding=utf-8
''' text multi label classification, when label is hierarchy
'''
import sys
import inspect
from collections import OrderedDict, defaultdict
from os import makedirs
import os
import json

import numpy as np
from sklearn.metrics import classification_report,f1_score
import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from shutil import rmtree

from config import config_func
from tf_utils.data import sequence_line_processing, data_line_processing,\
         json_line_processing,LineBasedDataset
from tf_utils.model import TextCnnLayer 
from utils.base import get_vocab
from utils.base import get_func_args
from utils import byteify
from nlp import load_w2v
from nlp.word2vec import save_w2v
config=None

class TextCNN(object):
    """
    cnn network for multi label text classification, especially label vocab is big.
    """
    def __init__(self, 
            sequence_length, 
            num_classes, 
            filter_sizes, 
            num_filters, 
            vocab_size=None,
            embedding_size=None, 
            l2_reg_lambda=0.0,
            freeze=False,
            exclusive=True,
            init_embedding=None,
            init_idf=None,
            class_weights=None,
            repr_mode='cnn',
            sampled_softmax=True,
            sess=None):
        args=get_func_args()
        for arg in args:
            setattr(self, arg, args[arg])

        self.build_inputs()
        self.build_embeddings()
        if repr_mode=='cnn':
            self.build_sent_cnn()
        elif repr_mode=='add':
            self.build_sent_add()
        elif repr_mode=='add+idf':
            self.build_sent_add_idf()
        self.build_loss()
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.summary_dir,sess.graph)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
    def build_inputs(self):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int64, [None, None, self.sequence_length], name="input_x")
        self.input_x_text_length = tf.placeholder(tf.int64, [None,None], name='input_x_text_length')
        self.input_x_num = tf.placeholder(tf.int64, [None], name='input_x_num')
        self.sampled_input_x =\
            tf.placeholder(tf.int64, [None, None, self.sequence_length], name="sampled_input_x")
        self.sampled_input_x_text_length =\
            tf.placeholder(tf.int64, [None,None], name='sampled_input_x_text_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size=tf.shape(self.input_x)[0]
        self.inputs = [ self.input_x, self.input_x_text_length, self.dropout_keep_prob ]
        self.topn = tf.placeholder(tf.int32, name="topn")

    def build_embeddings(self):
        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            if self.freeze and self.init_embedding is not None:
                self.emb = W = tf.Variable( self.init_embedding, name="W", trainable=False)
            else:
                self.emb = W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
            self.words_emb = tf.nn.embedding_lookup(W, self.input_x)
            self.words_emb = tf.expand_dims(self.words_emb, -1)
        # if use_add

    def build_sent_cnn(self):
        # reshape words_emb
        self.sent_vec = TextCnnLayer(self.sequence_length, config.vec_len, self.filter_sizes, self.num_filters)(self.words_emb)
        # Add dropout
        with tf.name_scope("dropout"):
            self.sent_vec = tf.nn.dropout(self.sent_vec, self.dropout_keep_prob)
        self.repr_len = sum(self.num_filters)
        # reshape back
        
    def build_sent_add(self):
        input_x_text_length = tf.reshape(self.input_x_text_length , [-1])
        self.mask = tf.cast(tf.sequence_mask(input_x_text_length, config.max_len), tf.float32)
        self.mask = tf.reshape(self.mask, [config.batch_size, len(config.langs), -1])
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,:,0],2)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        self.repr_len = config.vec_len
        #reshape back
        
    def build_sent_add_idf(self):
        self.mask = tf.cast(tf.sequence_mask(self.sequence_length, config.max_len), tf.float32)
        self.idf_x = tf.nn.embedding_lookup(self.idf, self.input_x)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0]\
                    *tf.expand_dims(self.idf_x,-1),1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        
    def build_loss(self):
        with tf.name_scope("dist_loss"):
            ave = tf.reduce_mean(self.sent_vec,1)
            dist = self.sent_vec - tf.expand_dims(ave,-2)
            dist_loss=tf.reduce_sum(tf.square(dist))
        with tf.name_scope("sampled_loss"):
            sampled_dist=self.sampled_sent_vec - tf.expand_dims(ave,-2)
            sampled_loss=tf.reduce_sum(tf.square(sampled_dist))
        self.loss=dist_loss+config.sampled_weight*sampled_loss
        tf.summary.scalar("loss", self.loss)
            
def train(sess, model, train_data, eval_data, batches, words=None, tags=None):
    sess.run(model.init)
    step=0
    best_eval_score=None
    for batch in range(config.batches):
        for k,inputs in enumerate(train_data):
            # if k>=150:break
            step+=1
            inputs = inputs + [config.dropout_ratio]
            fd=dict(zip(model.inputs, inputs))
            loss,_=sess.run([model.loss, model.train_op], feed_dict=fd)
            # print sess.run(model.out_labels, feed_dict = fd)
            print step, k, loss
            if batch>=1 and k==0:
                score=evaluate(sess,model,eval_data,tags)
                print 'EVULATION step={} score={}'.format(step,score)
                sys.stdout.flush()
                if True or not best_eval_score or score>best_eval_score:
                    print 'UPDATE MODEL...'
                    best_eval_score=score
                    model.saver.save(sess, config.model_output+'/model')
                    vectors=sess.run(model.emb, feed_dict=fd)
                    w2v=OrderedDict([(words[i],vectors[i]) for i in range(len(vectors))])
                    save_w2v(w2v, config.w2v_path)
                    
            if step % config.summary_steps == 0:
                summary = sess.run(model.merged, feed_dict=fd)
                model.train_writer.add_summary(summary, step)
    return None

def evaluate(sess, model, eval_data, target_names=None, reload=False):
    return 1
    if reload:
        print 'reload model...'
        model.saver.restore(sess, config.model_output)
    total_score, total_num=0,0
    print '-'*20+'start evaluate...'+'-'*20
    for k,inputs in enumerate(eval_data):
        fd=dict(zip(model.inputs, inputs+[1]))
        batch_num, score=\
            sess.run([model.batch_size, model.score], feed_dict = fd)
        total_num +=  batch_num
        total_score += score*batch_num
    score= total_score/total_num
    return -score

def predict(sess, model, inputs, tags_names):
    pass

def get_label_nums(train_data, tags=None):
    label_num=np.array([0,]*len(tags),dtype=np.float32)
    for sequence, _, labels in train_data:
        labels=np.sum(labels,axis=0)
        label_num+=labels
    return label_num

class json_line_processing_():
    def __init__(self, keys, line_processing):
        self.line_processing = line_processing
        self.keys=set(keys)
        self.size=line_processing.size

    def __call__(self, line):
        rval=[[] for i in range(self.size)]
        a=byteify(json.loads(line.strip()))
        if len(set(a.keys())&self.keys)!=len(self.keys):
            return None
        for key in a:
            if key in self.keys:
                data=self.line_processing(a[key])
                assert len(data)==len(rval)
                for i in range(len(data)):
                    rval[i].append(data[i])
        return rval
                
def main():
    if os.path.exists(config.summary_dir):
        rmtree(config.summary_dir)
    makedirs(config.summary_dir)
    words={k:word.strip() for k,word in enumerate(open(config.words_path)) if k<=config.vocab_size}
    text_processing = sequence_line_processing(words, max_len=config.max_len, return_length=True)
    vocab_size=len(text_processing.vocab)
    line_processing = json_line_processing_( config.langs, text_processing)
    train_data = LineBasedDataset(config.train_data, line_processing, batch_size= config.batch_size)
    dev_data = LineBasedDataset(config.dev_data, line_processing, batch_size = config.batch_size)
    for inputs in dev_data:
        for inp in inputs:
            print '-'*30
            print inp.shape
            #print inp
        #print inp
        break

    #with tf.Session() as sess:
    with tf.Session(config=config.session_conf) as sess:
        with tf.name_scope(config.lang):
            model=TextCNN(config.max_len, None, [1], [1000], freeze=False, exclusive=False, 
                vocab_size=vocab_size, embedding_size=config.vec_len, sess=sess, repr_mode=config.mode)
        train(sess, model, train_data, dev_data, config.batches, 
                words=[words[i] for i in range(len(words))])

if __name__=='__main__':
    global config
    lang='zh'
    config = config_func(lang)
    main()
