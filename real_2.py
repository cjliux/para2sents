# -*- coding: utf-8 -*-
import os
import re
import sys
import utils
import numpy as np
from pprint import pprint
import json
from collections import defaultdict
from operator import itemgetter
import pickle as pkl
import itertools as itt
"""
    used for real data
        just need to specify the data source as 'data_path'
"""


def split_para(ps, eat=False):
    nps = []
    for i, p in enumerate(ps):
        p1,p2 = p[0],p[1]
        np1 = utils.split_real(p1)
        np2 = utils.split_real(p2)
        nps.append((np1, np2) + p[2:])
    return nps


def mysplit(ps):
    return split_para(ps, eat=False)

def myjoin(ps):
    return utils.join_para(ps, con1='\n', con2='\n')

def save_stats(ps, save_to):
    d = defaultdict(int)
    for p in ps:
        d[(len(p[0]),len(p[1]))] += 1
    d2l = sorted([(k[0],k[1],v) for k,v in d.items()], key=itemgetter(2), reverse=True)
    with open(save_to, 'w', encoding='utf8') as fd:
        for s in d2l:
            fd.write('%d,%d,%d\n'%s)

def task1_extr_lt2(ps):
    return list(itt.ifilter(lambda x: len(x[0]) <= 2 and len(x[1]) <= 2, ps))

def main(data_path, corpus, result):
    ps = mysplit(utils.read_para(*corpus, revr=False))
    with open(os.path.join(data_path,'ps.pkl'), 'wb') as fd:
        pkl.dump(ps,fd,pkl.HIGHEST_PROTOCOL)  
    #with open(os.path.join(data_path,'ps.pkl'), 'rb') as fd:
    #    ps = pkl.load(fd)
    utils.save_para(myjoin(ps), result, sep=True)
    save_stats(ps,os.path.join(data_path,'stats.csv'))
    
    #fps = task1_extr_lt2(ps)
    #with open(os.path.join(data_path,'ps_lt2.pkl'), 'wb') as fd:
    #    pkl.dump(fps,fd,pkl.HIGHEST_PROTOCOL)
    #utils.save_para(myjoin(fps), os.path.join(data_path,'sep_lt2.txt'), sep=True)
    #save_stats(fps, os.path.join(data_path,'stats_lt2.csv'))
    
    
if __name__=='__main__':
    data_path = './real_data/data8'
    corpus = [ 'seg_ch.txt', 'seg_en.txt' ]
    result = 'sep1.txt'
    
    assert len(corpus) == 2
    
    corpus = [ os.path.join(data_path, x) for x in corpus ]
    result = os.path.join(data_path, result)
    main(data_path, corpus, result)
    
