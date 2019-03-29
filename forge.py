# -*- coding: utf-8 -*-
import string
import numpy as np
from collections import Counter
from utils import read_para, save_para, fbeta
from match import dp_match, dp_merge, match_paras

"""
gen
"""
def get_tf(corpus, letters=string.ascii_lowercase):
    tf = {}
    with open(corpus, 'r') as fd:
        cd = Counter(string.lower(fd.read()))
        for k, v in cd.iteritems():
            if k in letters:
                tf[k] = v
    total = sum(tf.values())
    tf = {k:float(v)/total for k,v in tf.iteritems()}
    return tf

def gen_seq(tf, minL, maxL):
    L, p = np.random.choice(range(minL,maxL+1)), []
    for l in xrange(L):
        p.append(np.random.choice(a=tf.keys(), p=tf.values()))
    return p

def gen_para(p, minB, maxB):     
    idx, p1 = 0, []
    while idx < len(p):
        b = np.random.choice(range(minB, maxB+1))
        p1.append(''.join(p[idx:idx+b]))
        idx += b

    idx, p2 = 0, []
    while idx < len(p):
        b = np.random.choice(range(minB, maxB+1))
        p2.append(''.join(p[idx:idx+b]))
        idx += b
    
    return p1, p2

def make_forge(tf, minL=10, maxL=72, minB=3, maxB=8, N=50000):
    ps = []
    for i in xrange(N):    
        p = gen_seq(tf, minL, maxL)
        p = gen_para(p, minB, maxB)
        ps.append(p)
    return ps

def split_para(ps):
    nps = []
    for p in ps:
        t = (p[0].split(), p[1].split())
        nps.append(t + p[2:])
    return nps

def join_para(ps):
    nps = []
    for p in ps:
        t = (' '.join(p[0]), ' '.join(p[1]))
        nps.append(t + p[2:])
    return nps

"""
def
"""
def get_gt(p1, p2):
    gt = []
    i1, l1, i2, l2 = 0, 0, 0, 0
    while i1 < len(p1) or i2 < len(p2):
        if l1 < l2:
            l1 += len(p1[i1])
            i1 += 1
        elif l1 > l2:
            l2 += len(p2[i2])
            i2 += 1
        else: 
            gt.append(l1)
            l1 += len(p1[i1])
            l2 += len(p2[i2])
            i1 += 1
            i2 += 1
    return gt

def get_pr(p):
    pr = []
    l = 0
    for s in p:
        pr.append(l)
        l += len(s)
    return pr

def eval_match(fps, rps):
    from tqdm import tqdm
    
    tfs, nfs = 0., 0
    for fp, rp in tqdm(zip(fps, rps)):
        gt, pr = get_gt(*fp), get_pr(rp[0])
        fs = fbeta(gt, pr)
        tfs += fs
        nfs += 1
        
        if fs < 1:
            print fp[0], '\n',  fp[1], '\n', rp[0]
            print gt, '\n', pr
            print fbeta(gt, pr)
            break
    afs = tfs / nfs
    return afs
    
def live_test(tf, minL=15, maxL=80, minB=3, maxB=12):
    import time
    while True:
        fp = gen_seq(tf, minL, maxL)
        fp1, fp2 = gen_para(fp, minB, maxB)
        path = dp_match(fp1, fp2, fbeta, debug=0)
        match = dp_merge(fp1, fp2, path)
        rp1 = [s1 for s1, s2, s in match]
        #rp2 = [s2 for s1, s2, s in match]
        sc = [s for s1, s2, s, in match]
        print fp1, '\n', fp2, '\n', rp1
        
        gt, pr = get_gt(fp1, fp2), get_pr(rp1)
        print gt, '\n', pr, '\n', sc
        
        fs = fbeta(gt, pr)
        print fs
        if fs < 1: break
        time.sleep(1)


def main(corpus, forge, result):
    minL, maxL, minB, maxB = 10, 72, 3, 8
    
    tf = get_tf(corpus)
    #ps = make_forge(tf, minL, maxL, minB, maxB, N=50000)
    #save_para(join_para(ps), forge)
    
    #ps = split_para(read_para(forge))
    #rps = match_paras(ps, fbeta)
    #save_para(join_para(rps), result)
    
    #print eval_match(ps, rps)
    
    live_test(tf, minL, maxL, minB, maxB)

if __name__=='__main__':
    corpus = './forge_data/en_seg'
    forge = './forge_data/forge'
    result = './forge_data/match'
    main(corpus, forge, result)
