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

"""Multi-threaded word2vec mini-batched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does traditional minibatching.

The key ops used are:
* placeholder for feeding in tensors for each example.
* embedding_lookup for fetching rows from the embedding matrix.
* sigmoid_cross_entropy_with_logits to calculate the loss.
* GradientDescentOptimizer for optimizing the loss.
* skipgram custom op that does input processing.
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

from nlp.base import get_words
from tf_utils.train import standard_train
from tf_utils.data import LineBasedDataset
from data_utils import word2vec_line_processing

flags = tf.app.flags

flags.DEFINE_string("save_path", 'result', "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", '', "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("emb_dim", 300, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 5,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 20,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
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
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS

class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, session, id2word, counts):
    self.session = session
    self.id2word = id2word
    self.counts = counts
    self.vocab_size=len(id2word)
    self.build_graph()
    # self.save_vocab()
  
  def build_graph(self):
    """Build the graph for the full model."""
    # The training data. A text file.
    examples=tf.placeholder(tf.int64, [None])
    labels=tf.placeholder(tf.int64, [None])
    self.inputs=[examples, labels]
    true_logits, sampled_logits = self.forward(examples, labels)
    loss = self.nce_loss(true_logits, sampled_logits)
    tf.summary.scalar("NCE loss", loss)
    self.loss = loss
    self.optimize(loss)

    # Properly initialize all variables.
    self.init = tf.global_variables_initializer()
    tf.summary.scalar("loss", self.loss)
    self.step_summaries = tf.summary.merge_all()
    self.saver = tf.train.Saver()

  def forward(self, examples, labels):
    """Build the graph for the forward pass."""

    # Declare all variables we need.
    # Embedding: [vocab_size, emb_dim]
    init_width = 0.5 / FLAGS.emb_dim
    emb = tf.Variable(
        tf.random_uniform(
            [self.vocab_size, FLAGS.emb_dim], -init_width, init_width),
        name="emb")
    self._emb = emb

    # Softmax weight: [vocab_size, emb_dim]. Transposed.
    sm_w_t = tf.Variable(
        tf.zeros([self.vocab_size, FLAGS.emb_dim]),
        name="sm_w_t")

    # Softmax bias: [vocab_size].
    sm_b = tf.Variable(tf.zeros([self.vocab_size]), name="sm_b")

    # Global step: scalar, i.e., shape [].
    self.global_step = tf.Variable(0, name="global_step")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [FLAGS.batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=FLAGS.num_neg_samples,
        unique=True,
        range_max=self.vocab_size-1,
        distortion=0.75,
        unigrams=self.counts))

    # Embeddings for examples: [batch_size, emb_dim]
    example_emb = tf.nn.embedding_lookup(emb, examples)

    # Weights for labels: [batch_size, emb_dim]
    true_w = tf.nn.embedding_lookup(sm_w_t, labels)
    # Biases for labels: [batch_size, 1]
    true_b = tf.nn.embedding_lookup(sm_b, labels)

    # Weights for sampled ids: [num_sampled, emb_dim]
    sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
    # Biases for sampled ids: [num_sampled, 1]
    sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

    # True logits: [batch_size, 1]
    true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_b, [FLAGS.num_neg_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_w,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits

  def nce_loss(self, true_logits, sampled_logits):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / FLAGS.batch_size
    return nce_loss_tensor

  def optimize(self, loss):
    """Build the graph to optimize the loss function."""

    # Optimizer nodes.
    # Linear learning rate decay.
    # words_to_train = float(100000)
    # lr = FLAGS.learning_rate * tf.maximum(
    #      0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
    # self._lr = lr
    lr=0.001
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss,
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
    self.train_op = train



  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    with open(os.path.join(FLAGS.save_path, "vocab.txt"), "w") as f:
      for i in xrange(self.vocab_size):
        vocab_word = tf.compat.as_text(FLAGS.vocab_words[i]).encode("utf-8")
        f.write("%s %d\n" % (vocab_word,
                             FLAGS.vocab_counts[i]))

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

def main(_):
    words_count = get_words([FLAGS.train_data], FLAGS.min_count)
    words_count = sorted(words_count.iteritems(), key=lambda x:x[1], reverse=True)
    id2words={i:word for i,(word,count) in enumerate(words_count)}
    counts=[count for word,count in words_count]
    line_processing = word2vec_line_processing(id2words, FLAGS.window_size)
    train_data=LineBasedDataset([FLAGS.train_data], line_processing, len=2, batch_size=FLAGS.batch_size)
    dev_data=None
    test_datas=None
    with tf.Session() as sess:
        model=Word2Vec(sess, id2words, counts)
        standard_train(
            sess, model,
            train_data, dev_data=[], test_datas=[],
            train_op=model.train_op,
            summary_dir='./logs',model_dir='./model',model_name='model',
            step_summary_op=model.step_summaries, epoch_summary_op=None, summary_steps=5, summary_epoch=1,
            init=True,
            epoch_num=20,
            eval_func=None, score_key='f1', test_func=None)# evaluate_func(sess, model, dev_data) return scalar
    standard_train()    

if __name__ == "__main__":
  tf.app.run()
