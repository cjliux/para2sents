# -*- coding: utf-8 -*-
import os
import re
from utils import is_zh
import string
from tqdm import tqdm
import cPickle as pkl
from collections import defaultdict, Counter

def w2v_voc2lex(data_dir='./real_data/test/', debug=0):
    voc = os.path.join(data_dir,'w2v/ch_w2v.voc')
    lex = os.path.join(data_dir,'/dict/ch_lex')

    if debug >= 1:
        fd = open(os.path.join(data_dir,'dict/ch_lex1'), 'w')
    with open(voc, 'r') as fd1:
        with open(lex, 'w') as fd2:
            for l in fd1:
                l = l.strip().split()[0].decode('utf-8')
                if is_zh(l):
                    fd2.write(l.encode('utf-8') + '\n')
                elif debug >= 1:
                    fd.write(l.encode('utf-8') + '\n')
    if debug >= 1: fd.close()

    voc = os.path.join(data_dir,'w2v/en_w2v.voc')
    lex = os.path.join(data_dir,'dict/en_lex')
 
    if debug >= 1:
        fd = open(os.path.join(data_dir,'dict/en_lex1'), 'w')
    with open(voc, 'r') as fd1:
        with open(lex, 'w') as fd2:
            for l in fd1:
                l = l.strip().split()[0].decode('utf-8')
                if (any([x in string.lowercase for x in l]) and 
                    all([x in string.lowercase+string.punctuation for x in l])):
                    fd2.write(l.encode('utf-8') + '\n')
                elif debug >= 1:
                    fd.write(l.encode('utf-8') + '\n')
    if debug >= 1: fd.close()

def algn2dict(algn):
    D = defaultdict(Counter)
    with open(algn, 'r') as fd:
        lines = fd.readlines()
        t = tqdm(total=len(lines)/3)
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip().decode('utf-8')
            if not line.startswith('#'):
                break
            l1 = lines[idx+1].strip().decode('utf-8').split()
            l2 = lines[idx+2].strip().decode('utf-8')
            
            s = 0
            for it in re.finditer(ur'\({(.*?)}\)', l2):
                w = l2[s:it.start()].strip()
                if w == 'NULL': continue
                a = Counter([ l1[int(i)-1] for i in it.group(1).split()])
                D[w] += a
                s = it.end()
            
            idx += 3
            t.update()
        t.close()
    
    nD = {}
    for k, c in tqdm(D.items()):
        nD[k] = dict(c)
    return nD

def algn2dict2(algn1, algn2):
    D = defaultdict(int)
    fd1,fd2 = open(algn1, 'r'), open(algn2, 'r')
    lns1, lns2, idx = fd1.readlines(), fd2.readlines(), 0
    t = tqdm(total=len(lns1)/3)
    while idx < len(lns1):
        line = lns1[idx].strip().decode('utf-8')
        if not line.startswith('#'):
            break
        
        s1, i1, a1 = 0, 0, set()
        l11s = lns1[idx+1].strip().decode('utf-8').split()
        l12 = lns1[idx+2].strip().decode('utf-8')
        for it in re.finditer(ur'\({(.*?)}\)', l12):
            w = l12[s1:it.start()].strip()
            if w == 'NULL': continue
            a1.update([ (i1, int(i)-1) for i in it.group(1).split()])
            s1 = it.end()
            i1 += 1
        s2, i2, a2 = 0, 0, set()
        l21s = lns2[idx+1].strip().decode('utf-8').split()
        l22 = lns2[idx+2].strip().decode('utf-8')
        for it in re.finditer(ur'\({(.*?)}\)', l22):
            w = l22[s2:it.start()].strip()
            if w == 'NULL': continue
            a2.update([(int(i)-1, i2) for i in it.group(1).split()])
            s2 = it.end()
            i2 += 1
        a1.intersection_update(a2)
        
        # co
        for w1, w2 in a1:
            D[(l21s[w1], l11s[w2])] += 1
        idx += 3
        
        t.update()
    t.close()
    return D
                
def make_giza_dict(data_dir='./real_data/test', debug=0):
    ce_algn = os.path.join(data_dir,'giza/c2e.A3.final')
    ec_algn = os.path.join(data_dir,'giza/e2c.A3.final')
    ce_dict = os.path.join(data_dir,'dict/fdict2')
    D = algn2dict2(ce_algn, ec_algn)
    with open('D.pkl', 'w') as fd:
        pkl.dump(D, fd, pkl.HIGHEST_PROTOCOL)
    with open(ce_dict, 'w') as fd:
        for p, f in tqdm(D.items()):
            l = p[0] + ' ' + p[1] + ' ' + str(f)
            fd.write(l.encode('utf-8') + '\n')
    
if __name__=='__main__':
    make_giza_dict()
    