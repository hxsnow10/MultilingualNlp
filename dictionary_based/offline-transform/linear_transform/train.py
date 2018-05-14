# encoding =utf-8
import subprocess
cmd="./align.py -s /opt/xia.hong/data/wiki_data/vec/{sl}_vec.txt -t /opt/xia.hong/data/wiki_data/vec/en_vec.txt -m /opt/xia.hong/data/wiki_data/all_synonym -o result/{sl}_vec.txt -sl {sl} -tl en -log logs/{sl}"
cmd="./align.py -s /opt/xia.hong/data/wiki_data/vec/{sl}_vec.txt -t /opt/xia.hong/data/word_vectors/word2vec/GoogleNews-vectors-negative300.txt -m /opt/xia.hong/data/wiki_data/all_synonym -o result/{sl}_vec.txt -sl {sl} -tl en -log logs/{sl}"
childs=[]
#for sl in  ['zh']:
for sl in  ['zh','de','ar','fr','it','fa','es','pt']:
    cmd_=cmd.format(sl=sl)
    print cmd_
    childs.append(subprocess.Popen(cmd_.split()))
for child in childs:
    child.wait()
    
