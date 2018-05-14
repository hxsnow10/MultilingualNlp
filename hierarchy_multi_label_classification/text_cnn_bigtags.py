# encoding=utf-8
import sys
import os
from os import makedirs
from shutil import rmtree
import json

from itertools import islice
import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
import numpy as np
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score
from collections import OrderedDict

from nlp import load_w2v
from nlp.base import get_words
from tf_utils.data import *
from tf_utils.model import multi_filter_sizes_cnn 
from utils.base import get_vocab
from utils.base import get_func_args

from config import config_func

config=None

class TextClassifier(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,
            num_classes,
            vocab_size=None,
            init_embedding=None,
            init_idf=None,
            class_weights=None,
            emb_name=None,
            reuse=None,
            mode='train',
            name_scope=None,
            configp=None):
        # 为了dynamic_model时候, 那里的config能替换这个函数的config
        # 如果不加这一段，dynamic_model中全局config被更新了，这里的config还是None
        if configp:
            global config
            config=configp
        
        args=get_func_args()
        for arg in args:
            setattr(self, arg, args[arg])
        
        with tf.name_scope(self.name_scope):
            self.build_inputs()
            self.build_embeddings()
            if config.text_repr=='cnn':
                self.build_sent_cnn()
            elif config.text_repr=='add':
                self.build_sent_add()
            elif config.text_repr=='add+idf':
                self.build_sent_add_idf()
            self.build_noexclusive_sampled_outputs(self.sent_vec, self.num_classes)
            
            if mode=='train':
                global_step = tf.Variable(0, trainable=False, name='global_step')
                starter_learning_rate = config.learning_rate
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, config.step_decay, 0.50, staircase=True)
                tf.summary.scalar("learning_rate", learning_rate)
                # tf.train.GradientDescentOptimizer(learning_rate).minimize
                self.step_summaries = tf.summary.merge_all() 
                self.train_op = tf.train.AdamOptimizer(learning_rate, name="adam_{}".format(emb_name)).minimize(self.loss, global_step=global_step)
        self.init = tf.global_variables_initializer()
        self.all_vars=list(set(
            (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)+
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="share")+
            [x for x in tf.global_variables() if x.name==self.emb_name+":0"])))
        self.train_vars=[x for x in self.all_vars if x in tf.trainable_variables()]
        self.all_saver=tf.train.Saver(self.all_vars)
        self.train_saver = tf.train.Saver(self.train_vars)
        print 'ALL VAR:\n\t', '\n\t'.join(str(x) for x in self.all_saver._var_list)
        print 'TRAIN VAR:\n\t', '\n\t'.join(str(x) for x in self.train_saver._var_list)
        print 'INPUTS:\n\t', '\n\t'.join(str(x) for x in self.inputs)
        print 'OUTPUTS:\n\t', '\n\t'.join(str(x) for x in self.outputs)

        
    def build_inputs(self):
        # Placeholders for input, output and dropout
        print id(config)
        self.input_x = tf.placeholder(tf.int64, [None, config.sen_len], name="input_x")
        self.input_sequence_length = tf.placeholder(tf.int64, [None], name="input_l")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        if self.mode=='train':
            self.input_y = tf.placeholder(tf.int64, [None, config.max_tags], name="input_y")
            self.input_y_prob = tf.placeholder(tf.float32, [None, config.max_tags], name="input_y_prob")
            self.class_weights=tf.constant(self.class_weights)
            self.inputs = [ self.input_y, self.input_y_prob, self.input_x, self.input_sequence_length, self.dropout_keep_prob ]
        else:
            self.inputs = [ self.input_x, self.input_sequence_length, self.dropout_keep_prob ]
        self.batch_size=tf.shape(self.input_x)[0]
        self.outputs = []
        if config.text_repr=='add+idf':
            self.idf = tf.Variable(self.init_idf, dtype=tf.float32, name='idf', trainable=False)

    def build_embeddings(self):
        # Embedding layer
        # tf not allowed init value of any tensor >2G
        if self.init_embedding is not None:
            init_emb=tf.constant(self.init_embedding, dtype=tf.float16)
            W = tf.get_variable(self.emb_name, initializer=init_emb, trainable=False)
        else:
            W = tf.get_variable(self.emb_name, shape=[self.vocab_size, config.vec_len], trainable=True)
        self.words_emb = tf.cast(tf.nn.embedding_lookup(W, self.input_x), tf.float32)
        self.words_emb = tf.expand_dims(self.words_emb, -1)
    
    def build_rnn(self):
        pass

    def build_sent_cnn(self):
        with tf.variable_scope("share"):
            self.sent_vec = multi_filter_sizes_cnn(self.words_emb, config.sen_len, config.vec_len, config.filter_sizes, config.filter_nums, name='cnn', reuse=self.reuse)
        with tf.name_scope("dropout"):
            self.sent_vec = tf.nn.dropout(self.sent_vec, self.dropout_keep_prob)
        self.repr_len = sum(config.filter_nums)
        
    def build_sent_add(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.sen_len), tf.float32)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0],1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        
    def build_sent_add_idf(self):
        self.mask = tf.cast(tf.sequence_mask(self.input_sequence_length, config.sen_len), tf.float32)
        self.idf_x = tf.nn.embedding_lookup(self.idf, self.input_x)
        self.sent_vec = tf.reduce_sum(tf.expand_dims(self.mask,-1)*self.words_emb[:,:,:,0]\
                    *tf.expand_dims(self.idf_x,-1),1)
        self.sent_vec = tf.nn.l2_normalize(self.sent_vec, dim = -1)
        self.repr_len = congig.vec_len
        
    def build_exclusive_ouputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            with tf.variable_scope("share"):
                self.scores = tf.layers.dense(inputs, num_classes, name="dense", reuse=self.reuse)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.outputs.append(self.predictions)
        if self.mode!='train':return

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.l2_loss=0#TODO
            self.loss = tf.reduce_mean(losses) + config.l2_lambda * self.l2_loss
        tf.summary.scalar("loss", self.loss)    
        self.outputs.append(self.loss)
        
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        # tf.summary.scalar("accuracy", self.accuracy)    

    def build_nonexclusive_outputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            with tf.variable_scope("share"):
                self.scores = tf.layers.dense(inputs, num_classes*2, name="dense", reuse=self.reuse)
            self.scores = tf.reshape(self.scores, [self.batch_size, num_classes, 2])
            self.predictions = tf.argmax(self.scores, -1, name="predictions")
            self.outputs.append(self.predictions)
        if self.mode!='train':return

        with tf.name_scope("loss"):
            input_y = tf.one_hot(self.input_y,depth=2,axis=-1)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=input_y)
            self.loss = tf.reduce_mean(losses*tf.expand_dims(self.class_weights,0))
        tf.summary.scalar("loss", self.loss)    
        self.outputs.append(self.loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.class_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), axis = 0, name="class_accuracy")
        # tf.summary.scalar("accuracy", self.accuracy)    
            
    def build_noexclusive_sampled_outputs(self, inputs, num_classes):
        with tf.name_scope("output"):
            #weights = tf.get_variable("weights",shape=[num_classes, sum(self.num_filters)],dtype=tf.float32,\
            #        initializer=tf.contrib.layers.xavier_initializer())
            #biases = tf.get_variable("biases",shape=[num_classes], dtype=tf.float32,\
            #        initializer=tf.constant_initializer(0.2))
            weights = tf.get_variable("weights",shape=[num_classes, self.repr_len],dtype=tf.float32)
            biases = tf.get_variable("biases",shape=[num_classes], dtype=tf.float32)
            tf.summary.histogram('weights',weights)
            tf.summary.histogram('biases',biases)

        # as class number is big, use sampled softmax instead dense layer+softmax
        with tf.name_scope("loss"):
            tags_prob = tf.pad(self.input_y_prob,[[0,0],[0,config.num_sampled]])
            out_logits, out_labels= _compute_sampled_logits( weights, biases, self.input_y, inputs,\
                    config.num_sampled, num_classes, num_true= config.max_tags )
            # TODO:check out_labels keep order with inpuy
            weighted_out_labels = out_labels * tags_prob*config.max_tags
            # self.out_labels = weighted_out_labels
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out_logits, labels=weighted_out_labels))
        
        with tf.name_scope("outputs"):
            logits = tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)
            self.output_values, self.ouput_indexs = tf.nn.top_k(logits, config.topn)

        with tf.name_scope("score"):
            self.score = self.loss/tf.cast(self.batch_size, tf.float32)
            #self.accuracy = tf.reduce_sum( self.top_prob )
   
        tf.summary.scalar('loss', self.loss)

def train(sess, source_model, target_model, 
        train_data, dev_data, test_datas, tags,
        summary_writers, class_summary_writers=None):
    sess.run(source_model.init)
    step=0
    best_score=0
    for batch in range(config.epoch_num):
        for k,inputs in enumerate(train_data):
            step+=1
            fd=dict(zip(source_model.inputs, inputs+[config.dropout_ratio]))
            if step%config.summary_steps!=0:
                loss,_=sess.run([source_model.loss, source_model.train_op], feed_dict=fd)
            else:
                loss,_,summary=\
                    sess.run([source_model.loss, source_model.train_op, source_model.step_summaries], feed_dict=fd)
                summary_writers['train'].add_summary(summary, step)
            print "batch={}\tstep={}\tglobal_step={}\tloss={}".format(batch, k, step ,loss)
            # eval every batch
            if k==0 and batch>=1:
                _,train_data_metrics = evaluate(sess,source_model,train_data,tags)
                score,dev_data_metrics = evaluate(sess,source_model,dev_data,tags)
                test_data_metricss = [evaluate(sess,target_model,test_data,tags)[1]
                    for test_data in test_datas]
                def add_summary(writer, metric, step):
                    for name,value in metric.iteritems():
                        summary = tf.Summary(value=[                         
                            tf.Summary.Value(tag=name, simple_value=value),   
                            ])
                        writer.add_summary(summary, global_step=step)
                add_summary(summary_writers['train'], train_data_metrics, step)
                add_summary(summary_writers['dev'], dev_data_metrics, step)
                for i in range(len(test_data_metricss)):
                        add_summary(summary_writers['test_{}'.format(i)], test_data_metricss[i], step)
                # add_summary(summary_writers['test-2'], test_data_metricss[1], step)
                
                
                if score>best_score:
                    best_score=score
                    # source_model.train_saver.save(sess, config.model_path, global_step=step)
                    source_model.train_saver.save(sess, config.model_path+'_'+source_model.name_scope, 
                            global_step=step)
                    source_model.all_saver.save(sess, config.model_path+'_'+source_model.name_scope+'_all')

def evaluate(sess, model, eval_data, target_names=None,restore=False):
    #model.saver.restore(sess, config.model_output)
    if restore:
        print 'reload model...'
        model.saver.restore(sess, config.model_output)
    total_y,total_predict_y = [], []
    print 'start evaluate...'
    all_rel, all_acc=[], []
    def clc(A,B):
        acc,recall=0,0
        d_A={y:v for y,v in zip(A[0].tolist(),A[1].tolist())}
        d_B={y:v for y,v in zip(B[0].tolist(),B[1].tolist())}
        for y in set(d_A.keys())&set(d_B.keys()):
            acc+=d_A[y]
            recall+=d_B[y]
        return acc,recall

    for inputs in eval_data:
        fd=dict(zip(model.inputs, inputs+[1]))
        py,pyw=\
            sess.run([model.ouput_indexs, model.output_values], feed_dict = fd)
        y, yw=inputs[0], inputs[1]
        for i in range(py.shape[0]):
            acc,rel=clc((py[i],pyw[i]),(y[i],yw[i]))
            all_rel.append(rel)
            all_acc.append(acc)

                
        #total_predict_y = total_predict_y + [predict_y]
    return 0,{"recall":sum(all_rel)/len(all_rel),"precition":sum(all_acc)/len(all_acc)}

def get_label_nums(train_data, tags=None):
    label_num=np.array([0.01,]*len(tags),dtype=np.float32)
    return label_num
    for labels,labels_weights,_,_ in train_data:
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label_num[int(labels[i][j])]+=labels_weights[i,j]
    return label_num

def main():
    def check_dir(dir_path, ask_for_del):
        if os.path.exists(dir_path):
            y=''
            if ask_for_del:
                y=raw_input('new empty {}? y/n:'.format(dir_path))
            if y.strip()=='y' or not ask_for_del:
                rmtree(dir_path)
            else:
                print('use a clean summary_dir')
                quit()
        makedirs(dir_path)
        oo=open(os.path.join(dir_path,'config.txt'),'w')
        d={}
        for name in dir(config):
            if '__' in name:continue
            d[name]=getattr(config,name)
        oo.write(json.dumps(d,ensure_ascii=False))
    check_dir(config.summary_dir, config.ask_for_del)
    check_dir(config.model_dir, config.ask_for_del)
    
    words={k:word.strip() for k,word in enumerate(islice(open(config.words_path),config.max_vocab_size))}
    tags={k:tag.strip() for k,tag in enumerate(open(config.tags_path))}
    target_processing = sequence_line_processing(tags, max_len=config.max_tags, return_length=False)
    tags_size=len(target_processing.vocab)
    weight_processing = data_line_processing(max_len=config.max_tags)
    text_processing = sequence_line_processing(words, max_len=config.sen_len, return_length=True)
    vocab_size=len(text_processing.vocab)
    # datas
    line_processing = json_line_processing(
            OrderedDict((("tags",target_processing),("weights",weight_processing),("text",text_processing))))
    train_data = LineBasedDataset(config.train_data_path, line_processing, batch_size= config.batch_size) 
    dev_data = LineBasedDataset(config.dev_data_path, line_processing, batch_size = config.batch_size)
    test_datas = [LineBasedDataset(path, line_processing, batch_size = config.batch_size)
        for path in config.test_data_paths]
    
    # show shape
    for k,inputs in enumerate(train_data):
        print '-'*20,'batch ',k,'-'*20
        for inp in inputs:
            print inp.shape
        if k>=3:break
    
    # compute class weights for class unbalanced
    class_nums=get_label_nums(train_data, tags)
    class_weights=class_nums/np.sum(class_nums)*len(class_nums)
    print 'TRAIN CLASSES=\t',tags.values()
    print 'TRAIN CLASS_NUMS=\t',class_nums
    print 'TRAIN CLASS_WEIGHTS=\t',class_weights
     
    with tf.Session(config=config.session_conf) as sess:
        # use tf.name_scope to manager variable_names
        model=TextClassifier(
            num_classes=len(tags), 
            vocab_size=vocab_size, 
            class_weights=class_weights,
            emb_name='emb',
            reuse=False,
            mode='train',
            name_scope="train")

        # summary writers for diiferent branch/class
        summary_writers = {
            sub_path:tf.summary.FileWriter(os.path.join(config.summary_dir,sub_path), flush_secs=5)
                for sub_path in ['train','dev']+["test_{}".format(i) for i in range(len(test_datas))]}
        
        train(sess, model, model,
                train_data, dev_data, test_datas, 
                tags=tags.values(),
                summary_writers=summary_writers)

if __name__=='__main__':
    global config
    config = config_func('de')
    main()
