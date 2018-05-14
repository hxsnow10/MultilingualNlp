# encoding=utf-8
import json

from utils import byteify
from collections import defaultdict

from config import config_func
config=config_func('zh')

def get_vocab(min_count):
    files=[config.train_data, config.dev_data]
    count=defaultdict(int)
    for file_path in files:
        ii=open(file_path)
        for line in ii:
            a=byteify(json.loads(line.strip()))
            for lang in a:
                if lang not in config.langs:continue
                text = a[lang]
                for word in text.split():
                    count[word]+=1
    count=sorted([x for x in count.iteritems() if x[1]>min_count], key=lambda z:z[1], reverse=True)
    oo=open(config.words_path,'w')
    for w,c in count:
        oo.write(w+'\n')

get_vocab(10)

            

