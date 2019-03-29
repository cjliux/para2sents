# -*- coding: utf-8 -*-
import os
import re
import sys
import numpy as np
from utils import read_para, save_para, split_para, join_para
from match import match_para

def mysplit(ps):
    return split_para(ps, sre1=ur'[。]', sre2=ur'[.]', eat=False)

def myjoin(ps):
    return join_para(ps, con1='\t', con2='\t')


def main(data_dir, corpus, result):
    ps = mysplit(read_para(*corpus))
    print ps[2]
    #save_para(myjoin(ps[:100]), *result)
    
    
if __name__=='__main__':
    #data_dir = sys.argv[1]
    #corpus = sys.argv[2].split(',')
    data_dir = './real_data/test'
    corpus=['ch_seg', 'en_seg']
    
    assert len(corpus) == 2
    
    corpus = [os.path.join(data_dir, x) for x in corpus]
    result = [x + '.out' for x in corpus]
    main(data_dir, corpus, result)
