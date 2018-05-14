# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
from threadsafe_generator import threadsafe_generator
from utils.wraps import count_time,ifrepeat
MAXLEN=4

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 5,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 20,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 6,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 10,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-4,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

flags.DEFINE_string("bilingual_data", "", 
                "bilingual data path, each line contains 2 item list split by <bfd/split>"
                "items list a list of item, each is like word/number")


FLAGS = flags.FLAGS



class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data

    self.bilingual_data = FLAGS.bilingual_data

class synonym_dataset():
    def __init__(self, data_path, _word2id, batch_size=100, max_len=10,pre_compute=False):
        self.data_path=data_path
        self._word2id=_word2id
        self.batch_size=batch_size
        self.finished=False
        self.max_len=max_len
        self.full=None
        start=time.time()
        if pre_compute:
            self.full=list(self.epoch_data(1))
            print('pre compute time', time.time()-start)
    
    def padding(self, input):
        s=np.array([0,]*self.max_len, dtype=np.int32)
        l=min(len(input), self.max_len)
        s[:l]=np.array(input[:l],dtype=np.int32)
        return s

    def processing_line(self, line):
        parts=line.strip().split('/')
        parts=[[self._word2id[word] for word in part.split(' ') if word in self._word2id]\
                for part in parts if len(part)<=4]
        parts=[part for part in parts if part]
        lengths=[len(part) for part in parts]
        inputs=[self.padding(part) for part in parts]
        samples=[]
        for i in range(len(lengths)):
            for j in range(len(lengths)):
                if i==j:continue
                samples.append([inputs[i], inputs[j], lengths[i], lengths[j]])
                #print([inputs[i], inputs[j], lengths[i], lengths[j]])
                #raw_input('xxxxx')
        return samples
    
    @threadsafe_generator 
    #@profile
    def epoch_data(self, epoch=10):
        if self.full:
            for i in range(epoch):
                for batch in self.full:
                    yield batch
            return
        self.finished=False
        start=time.time()
        tmp=[[],]*4
        #batch_data=
        processed=0
        for k in range(epoch):
            ii=open(self.data_path, 'r')
            s=0
            time1,time2=0,0
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
                        batch=[np.array(ele, dtype=np.int32) for ele in batch]
                        yield batch
                        processed+=self.batch_size
                        tmp=[tmp[j][self.batch_size:] for j in range(len(tmp))]
                time2+=(time.time()-sstart)
        while len(tmp[0])>0:
            l=min(self.batch_size,len(tmp[0]))
            batch=[tmp[j][:l] for j in range(len(tmp))]
            batch=[np.array(ele, dtype=np.int32) for ele in batch]
            yield batch
            tmp=[tmp[j][l:] for j in range(len(tmp))]
        self.finished=True

class bidataset():
    
    def __init__(self, data_path, _word2id, batch_size, split='/', window_size=5):
        self._word2id=_word2id
        self.batch_size=batch_size
        self.batch_tmp=[]
        self.data_path = data_path
        self.split = split
        self.window_size = 5
        self.finished=False

    def update_tmp(self):
        batches=int(len(self.examples_tmp)/self.batch_size)
        #print(len(self.examples_tmp))
        #print(len(self.labels_tmp))
        assert len(self.examples_tmp) == len(self.labels_tmp)
        for i in range(batches):
            ebatch=np.array(self.examples_tmp[i*self.batch_size:(i+1)*self.batch_size])
            lbatch=np.array(self.labels_tmp[i*self.batch_size:(i+1)*self.batch_size])
            self.batch_tmp.append((ebatch,lbatch))
        self.examples_tmp=self.examples_tmp[batches*self.batch_size:]
        self.labels_tmp=self.labels_tmp[batches*self.batch_size:]
    
    @threadsafe_generator 
    def epoch_data(self, epochs=1):
        self.finished=False
        for i in range(epochs):
            ii=open(self.data_path,'r')
            self.examples_tmp, self.labels_tmp = [], []
            for line in ii:
                #print('before line {}.{}'.format(len(self.examples_tmp), len(self.labels_tmp)))
                examples, labels=self.processline(line)
                self.examples_tmp+=examples
                self.labels_tmp+=labels
                if len(self.examples_tmp)>=1000*self.batch_size:
                    self.update_tmp()
                    for examples, labels in self.batch_tmp:
                        yield examples, labels
            self.update_tmp()
            for examples, labels in self.batch_tmp:
                yield examples, labels
        self.finished=True
        return

    def processline(self, line):
        rval=[]
        words=line.split()
        words=[[t for t in w.split(self.split) if t] for w in words]
        for i in range(len(words)):
            for j in range(max(0,i-self.window_size), min(len(words),i+self.window_size+1), 1):
                if len(words[i])<=1 and len(words[j])<=1:continue
                for w1s in words[i]:
                    for w2s in words[j]:
                        w1s=w1s.split('_')
                        w2s=w2s.split('_')
                        for w1 in w1s:
                            for w2 in w2s:
                                if w1 in self._word2id and w2 in self._word2id:
                                    rval.append((self._word2id[w1],self._word2id[w2]))
        examples=[x[0] for x in rval]
        labels=[x[1] for x in rval]
        assert len(examples)==len(labels)
        return examples, labels

        
class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    gpu_session_conf = tf.ConfigProto(
      #device_count = {'CPU': 0, 'GPU':1}, 
      allow_soft_placement=True,
      log_device_placement=False,)
    self.gpu_session=tf.Session(config=gpu_session_conf)
    self._word2id = {}
    self._id2word = []
    print('build graph...')
    self.build_graph()
    print('build eval graph...')
    self.build_eval_graph()
    print('build synonym graph...')
    self.build_synonym_graph()
    print('save vocab...')
    self.save_vocab()
    print('init data...')
    self.init_data()
    #with tf.Session() as sess:
    tf.global_variables_initializer().run()
    self.saver = tf.train.Saver()


  
  def init_data(self):

    self.bidataset=bidataset(self._options.bilingual_data, self._word2id, self._options.batch_size)

  def read_analogies(self):
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.strip().lower().split(b" ")
        ids = [self._word2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    # The training data. A text file.
    print('data graph build')
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                    batch_size=opts.batch_size,
                                                    window_size=opts.window_size,
                                                    min_count=opts.min_count,
                                                    subsample=opts.subsample)
    self.examples = examples
    self.labels = labels
    print('data read...')
    #with tf.Session() as sess:
    (opts.vocab_words, opts.vocab_counts,
        opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    print('data read finished')
    opts.vocab_size = len(opts.vocab_words)
    print("vocab counts: ",opts.vocab_counts)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)
    
    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i
    
    
    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    with tf.variable_scope("emb"):
        w_in = tf.get_variable(
            shape=[opts.vocab_size,opts.emb_dim], 
            dtype=tf.float32,
            name="w_in")

        # Global step: scalar, i.e., shape [].
        w_out = tf.get_variable(
            shape=[opts.vocab_size,opts.emb_dim], 
            dtype=tf.float32,
            name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    self.lr=lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train_word2vec(w_in,
                                          w_out,
                                          examples,
                                          labels,
                                          lr,
                                          vocab_count=opts.vocab_counts.tolist(),
                                          num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.global_step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed

  def build_synonym_graph(self):
    def average(seq, length, W):
        seq=tf.nn.embedding_lookup(W, seq)
        mask=tf.expand_dims(tf.sequence_mask(length, MAXLEN, dtype=tf.float32),-1)
        ave=tf.reduce_sum(seq*mask, axis=1)/tf.expand_dims(tf.cast(length, tf.float32)+0.01,-1)
        return ave

    self.input1 = input1 = tf.placeholder(tf.int32, [None,MAXLEN])
    self.input2 = input2 = tf.placeholder(tf.int32, [None,MAXLEN])
    self.input_length1 = input_length1 = tf.placeholder(tf.int32, [None])
    self.input_length2 = input_length2 = tf.placeholder(tf.int32, [None])
    with tf.variable_scope("emb", reuse=True):
        w_in = tf.get_variable('w_in')
        w_out = tf.get_variable('w_out')

    ave1=average(input1, input_length1, w_in+w_out)
    ave2=average(input2, input_length2, w_in+w_out)
    
    def cosin_distance(a,b):
        normalize_a = tf.nn.l2_normalize(a,1)
        normalize_b = tf.nn.l2_normalize(b,1)
        loss=1 - tf.reduce_mean(tf.reduce_sum(tf.multiply(normalize_a,normalize_b),1))
        return loss
    self.synonym_loss = cosin_distance(ave1, ave2)
    #self.synonym_train = tf.train.GradientDescentOptimizer(0.001).minimize(self.synonym_loss)
    self.synonym_train = tf.train.AdamOptimizer(0.001).minimize(self.synonym_loss)

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
        f.write("%s %d\n" % (vocab_word,
                             opts.vocab_counts[i]))

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.

  @ifrepeat(30)
  @count_time('synonym process') 
  def _train_thread_body_synonym(self):
    self.synonym_dataset=synonym_dataset('small_synonym.tok', self._word2id, self._options.batch_size, max_len=MAXLEN,pre_compute=False)
    self.sdata=self.synonym_dataset.epoch_data(1)
    self.synonym_dataset.finished=False
    k=0
    sum_loss=0
    #if k==1:
    start=time.time()
    while not self.synonym_dataset.finished:
        k+=1
        try:
            input1, input2, input_length1, input_length2 = self.sdata.next()
            fd={self.input1:input1, self.input2:input2, self.input_length1:input_length1, self.input_length2:input_length2}
            loss,_ = self._session.run([self.synonym_loss, self.synonym_train], feed_dict=fd)
            sum_loss+=loss
            if k%50==0:
                print('synonym step={}, use time {}'.format(k,time.time()-start))
            sys.stdout.flush()
        except StopIteration:
            self.synonym_dataset.finished=True
            break
    print('sum_loss={}'.format(sum_loss))
    if not self.before_loss or abs(sum_loss-self.before_loss)/(abs(self.before_loss)+0.001)>=0.002:
        self.before_loss=loss
        return True
    else:
        self.before_loss=loss
        return False
        
  @count_time()    
  def _train_thread_body_multi(self):
    self.bdata=self.bidataset.epoch_data(1)
    k=0
    while not self.bidataset.finished:
        k=k+1
        examples, labels= self.bdata.next()
        fd = {self.examples:examples, self.labels:labels}
        #a,b=self._session.run([self.examples, self.labels])
        #print('{},{}'.format(a.shape, b.shape))
        #print('a[0]={},a[0]_len={}'.format(a[0], a[0].shape))
        #print('b[0]={},b[0]_len={}'.format(b[0], b[0].shape))
        _= self._session.run([self._train], feed_dict=fd)
        if k%100==0:
            print('bilingual step={}'.format(k))

  #@count_time('word2vec process')
  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      # a,b=self._session.run([self.examples, self.labels])
      #print('{},{}'.format(a.shape, b.shape))
      #print('a[0]={},a[0]_len={}'.format(a[0], a[0].shape))
      #print('b[0]={},b[0]_len={}'.format(b[0], b[0].shape))

      _, epoch = self._session.run([self._train, self._epoch])
      if epoch == initial_epoch+1:
        break
  
  def train(self):
    """Train the model."""
    print('start train')
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    self.before_loss=None
    for _ in xrange(1):
      t = threading.Thread(target=self._train_thread_body_synonym)
      t.start()
      workers.append(t)
    for t in workers:
      t.join()
    start=time.time()
    workers = []  
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)
    '''
      for _ in xrange(2):
      t = threading.Thread(target=self._train_thread_body_multi)
      t.start()
      workers.append(t)
    ''' 
    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(5)  # Reports our progress once a while.
      (epoch, step, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate),
            end="")
      sys.stdout.flush()
      if epoch == initial_epoch+1:
        print('finished word2vec processes', time.time()-start)
        break

    for t in workers:
      t.join()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval2(self):
    self.sdata=self.synonym_dataset.epoch_data(1)
    self.synonym_dataset.finished=False
    k=20
    loss=0
    while not self.synonym_dataset.finished: 
        try:
            input1, input2, input_length1, input_length2 = self.sdata.next()
            fd={self.input1:input1, self.input2:input2, self.input_length1:input_length1, self.input_length2:input_length2}
            loss += self._session.run(self.synonym_loss, feed_dict=fd)
        except StopIteration:
            self.synonym_dataset.finished=True
            break
    print('eval loss={}'.format(loss))
      
  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    try:
      total = self._analogy_questions.shape[0]
    except AttributeError as e:
      raise AttributeError("Need to read analogy questions.")

    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        print(c)
        break
    print("unknown")

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))


def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  session_conf = tf.ConfigProto(
      #intra_op_parallelism_threads=1,
      #inter_op_parallelism_threads=1,
      device_count = {'CPU': 24, 'GPU':0}, 
      allow_soft_placement=True,
      log_device_placement=False,) 
  with tf.Graph().as_default(), tf.Session(config=session_conf) as sess:
    #with tf.device("/cpu:0"):
    model = Word2Vec(opts, sess)
    model.read_analogies() # Read analogy questions
    print('MODEL INITED')
    print('START TRAIN')
    for _ in xrange(opts.epochs_to_train):
        model.train()  # Process one epoch
        #model.eval()  # Eval analogies.
        model.eval2()
        model.saver.save(sess, os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.global_step)
    # Perform a final save.
    if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy(b'france', b'paris', b'russia')
      # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
      _start_shell(locals())

def test_multi_data():
    word2id={}
    ii=open('tmp/vocab.txt','r')
    for line in ii:                               
        word,_=line.strip().split()
        word2id[word]=len(word2id)
    dataset = bidataset('zh.en.bi',word2id, 100)
    for examples, labels in dataset.epoch_data(): 
        #print('{}{}'.format(examples,labels))
        print('{}{}'.format(examples.shape, labels.shape))


if __name__ == "__main__":
  #test_multi_data()
  tf.app.run()
