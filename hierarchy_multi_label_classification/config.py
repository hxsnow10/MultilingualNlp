import os
import tensorflow as tf

def config_func(target_, source_='en'):
    '''config depends on some parameters
    '''
    class config():
        
        # not share
        base_dir='/opt/xia.hong/data/wiki_data/wiki_multi_label_text/zh_test'
        words_path='{}/words.txt'.format(base_dir)
        tags_path='{}/tags.txt'.format(base_dir)
        train_data_path=['{}/train.txt'.format(base_dir)]
        dev_data_path=['{}/dev.txt'.format(base_dir)]
        test_data_paths = []
        
        # output
        model_dir='./model'
        model_path=os.path.join(model_dir,'model')
        # here source_model=target_model except emb
        summary_dir='./log'
        
        # model
        max_tags=15
        max_vocab_size=200
        sen_len=45000
        vec_len=300
        text_repr='cnn'
        default_idf=7
        drop_idf=True
        drop_th=3
        filter_sizes=[1]
        filter_nums=[1000]
        dropout_ratio=1
        exclusive=False
        l2_lambda=0
        learning_rate=0.01
        step_decay=5000
        topn=10
        num_sampled=15
        
        # other
        ask_for_del=False
        epoch_num=240
        batch_size=5
        summary_steps=5
        session_conf = tf.ConfigProto(
              device_count = {'CPU': 12, 'GPU':0}, 
              allow_soft_placement=True,
              log_device_placement=False,)
        session_conf=None 
    return config
