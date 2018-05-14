# encoding=utf-8
''' use wiki_base to multi_tag sent
1) a dict provibe for ability to check whether
a title is in wiki_base
2) a func to multi_tag
*3) disambugation
'''
from utils import byteify
from nlp.tokenize import tok as en_tok
from nlp.tokenize import end_marks
from utils.mpwork import BatchedMpWork
from chineseSegment.segment import Segment
from collections import defaultdict
import re
import random

zhseg=Segment(user_dict_path=['zh/wiki_dict.txt'])
zh_tok=lambda sent:zhseg.cut(sent)
iszh=re.compile(u'[\u4e00-\u9fff]+', re.UNICODE)
def tok(sent):
    print sent
    if iszh.search(sent.decode('utf-8')):
        return zh_tok(sent)
    else:
        return en_tok(sent)

class MultiName_WikiBase(object):

    def __init__(self, multi_dict_path, use_tok=False, one_gram=True):
        ii=open(multi_dict_path,'r')
        self.synonym=defaultdict(set)
        for line in ii:
            try:
                line_,weight=line.strip().split('\t')
                weight=float(weight)
                line=line_
            except:
                pass
            if '/' not in line or not line.strip() or '(' in line :continue
            parts=line.strip().split('/')
            if one_gram:
                parts=[part for part in parts]
            # parts=[part.split(' ') for part in parts]
            if len(parts)<=1:continue
            for part in parts:
                self.synonym[part].update(parts)
        #self.autoA = self.buildA(base.keys())
        print 'multiname init'
        
    def buildA(self, keys):
        A = ahocorasick.Automaton()
        for key in keys:
            A.add_word(key,key)
        A.make_automaton()
        return A

    def word_process(self, word):
        return '/'.join(x.replace(' ','_') for x in self.base[word])

    # one gram
    def multi_name(self, article, p=0.2):
        s=random.random()
        if s>0.01:return []
        words=article.strip().split()
        rval=[]
        for word in set(words):
            if word in self.synonym:
                for word2 in self.synonym[word]:
                    s=random.random()
                    if s>p:continue
                    def chg(x):
                        if x!=word:return x
                        else:return word2
                    new=[chg(w) for w in words]
                    new=' '.join(new)
                    rval.append(new)
        return rval
'''
    def multi_name(self, article):
        d_res={}
        for end,word in list(self.autoA.iter(article)):
            begin=end-len(word)+1
            if begin not in d_res or len(word)>d_res[begin]:
                d_res[begin]=word
        l_res=sorted([(end-len(word),end, word) for end,word in d_res.iteritems()], key=lambda x:(x[0],-x[1]))
        s=''
        fend=0
        for begin,end,word in l_res:
            if len(begin)<len(s):continue
            fend=end
            s+=article[len(s):end]
            s+=self.word_process(word)
        s+=article[fend:]
        return s
'''

splits=set(end_marks)
def sent_split(line):
    rval=[]
    tmp=[]
    for x in line.strip().split(' '):
        tmp.append(x)
        if x in splits:
            rval.append(' '.join(tmp))
            tmp=[]
    rval.append(' '.join(tmp))
    return [x for x in rval if x]

def data(files):
    for file_ in files:
        ii=open(file_,'r')
        for line in ii:
            try:
                sents=sent_split(line)
                for sent in sents:
                    yield sent
            except Exception,e:
                print e
           
mt=MultiName_WikiBase('synonym.tok')
def process(line):
    return mt.multi_name(line)

class listener():
    def __init__(self, path):
        self.f=open(path,'w')
    def __call__(self, line):
        self.f.write(line.strip()+'\n')

def main():
    out=listener('all_tok_multi_from_en.txt')
    for line in data(['en_wiki_data.txt.tok']):
        rval=process(line)
        for line in rval:
            out(line)
    # BatchedMpWork(data(),process, listener(), workers=2)

main()
