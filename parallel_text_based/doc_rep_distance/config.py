import tensorflow as tf

def config_func(lang_):
    class config():
        lang=lang_
        max_tags=15
        max_len=10
        num_sampled=40
        model_output='./model'
        base_dir = '/opt/xia.hong/data/wiki_data/wiki_multi_label_text/text_align'
        # 100W sample
        train_data='{}/train.txt'.format(base_dir)
        dev_data='{}/dev.txt'.format(base_dir)
        words_path='{}/words.txt'.format(base_dir)
        
        batches=30
        batch_size=8
        evaluate_steps=5000
        vec_len=300
        dropout_ratio=1
        session_conf = tf.ConfigProto(
              device_count = {'CPU': 12, 'GPU':0},
              allow_soft_placement=True,
              log_device_placement=False,)
        session_conf.gpu_options.allow_growth=False
        l2_lambda=0
        learning_rate=0.001
        mode='add'
        summary_dir='log'
        summary_steps=10
        w2v_path='vec.txt'
        vocab_size=400000
        langs=['zh','en']
    return config
