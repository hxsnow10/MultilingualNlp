#encoding=utf-8
import tensorflow as tf

vec_len = 300
sen_len = 1
epochs = 20
eval_steps = 50000
save_path = "./model/model"
new_vec_path = "./model/new_vec.txt"
theta = 1
dtype = tf.float32
dtype2 = tf.float32
#originalw2v_path = 'glove.840B.300d.txt'
originalw2v_path = 'small_glove.txt'
synonym_path = 'synonym.tok'
batch_size = 200
learning_rate = 0.01
max_gradient_norm = 2
pre_steps=1000000
