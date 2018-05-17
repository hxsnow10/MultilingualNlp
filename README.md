MultilingualNlp
===============
目的1：利用对齐语料(以及单语语料)训练出多语表示。

目的2：利用多语表示做迁移学习。

## 多语表示

### Loss include 
* monolingual signal
    * word2vec
    * rnn language model
    * sentiment supervision
    * wikipedia label graph

* translation siganl
    * dictionary alignments
    * sentence alignments
    * doc alignments
    * label aligned doc/sent
    * knowledge graph alignments, like wikidata、conceptNet
    * implicit regularization of translation

### implementation
```
├── dictionary_based    基于词对齐的多语表示学习，最终获得多语词向量
│   ├── multi_tag           一个不优雅单但实现简单的思路: 通过替换词生成word2vec训练文本
│   ├── offline-transform   提前训练好不同语言的词向量，训练一个变换 
│   ├── online-joint-loss   在线学习，loss使用单语与相似词表示的距离。目前尚未实现，推荐。 
│   ├── retrofit            [ConceptNet]训练好一个语言的词向量，通过词对齐、词图谱获得自身语言的纠正、其他语言表示。
│   ├── word2vec-master     利用C版word2vec实现的多语词向量学习
│   │   ├── word2vec_freeze.c   输入包括之前训练的好的一个词向量，在新语料上增量训练，把之前的词向量freeze
│   │   ├── word2vec_synonym.c  输入包括一个同义词词典，使用同义词替代实时训练，速度还行，方法感觉不是很OK
│   └── word2vec-tf         利用tensorflow实现的多语词向量学习。
│       ├── data_utils.py       数据python接口
│       ├── pure_tf_word2vec.py 纯tf代码
│       ├── train_synonym.sh
│       ├── word2vec2.py        
│       ├── word2vec_optimized_multi.py     
│       ├── word2vec_optimized_synonym2.py
│       ├── word2vec_optimized_synonym.py
├── hierarchy_multi_label_classification    利用多语文章、标签对齐训练。wikipedia对齐标签(需要wikidata)
├── parallel_text_based     基于句子对齐的多语表示学习。
│   ├── bicvm               http://arxiv.org/abs/1404.4641.
```

### 评估
* analogy_test:               利用手工标注的词相似度对模型评估
* classification_test         http://git.baifendian.com/xia.hong/MultilingualClassification.git 

## 迁移学习
http://git.baifendian.com/xia.hong/MultilingualClassification.git
缺乏落地
