# encoding=utf-8


def main():
    ii=open('synonym.tok', 'r')
    oo=open('dict.txt', 'w')
    for line in ii:
        try:
            line_,weight=line.strip().split('\t')
            line=line_
        except:
            pass
        parts=line.strip().split('/')
        parts=[part for part in parts if ' ' not in part and '/' not in part]
        if len(parts)<=1:continue
        oo.write(' '.join(parts)+'\n')
main()
