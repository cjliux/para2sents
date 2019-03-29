# -*- coding: utf-8 -*-
import numpy as np

def deorder(p1, p2, f, debug=0):
    pass

def dp_match(p1, p2, f, debug=0):
    if debug >= 1:
        print 'matching:\n', ' '.join(p1), '\n', ' '.join(p2)
    
    n, m = len(p1), len(p2)
    bp1, bp2 = np.zeros((n,m)).astype(np.int), np.zeros((n,m)).astype(np.int)
    sc, nsc = np.zeros((n,m)), np.ones((n,m)).astype(np.int)
    # dp
    for i in range(n):
        for j in range(m):
            s1, s2 = ''.join(p1[:i+1]), ''.join(p2[:j+1])
            sc[i,j] = np.log(max(f(s1,s2),1e-6))
    if debug >= 2:
        print 'before:\n' + repr(sc)
    for i in xrange(n):
        for j in xrange(m):
            for ii in xrange(i,0,-1):
                for jj in xrange(j,0,-1):
                    s1, s2 = ''.join(p1[ii:i+1]), ''.join(p2[jj:j+1])
                    ns = nsc[ii-1,jj-1]
                    s = (np.log(max(f(s1,s2), 1e-6)) + ns*sc[ii-1,jj-1])/(ns+1)
                    if s > sc[i,j] or abs(s-sc[i,j]) < 1e-6 and ns+1 > nsc[i,j]:
                        sc[i,j], nsc[i,j], bp1[i,j], bp2[i,j] = s, ns+1, ii, jj
    if debug >= 2:
        print 'after:\n' + repr(sc)
    
    # backtrace
    btr = []
    i, j = n-1, m-1
    while i >= 0 and j >= 0:
        ii, jj = bp1[i,j], bp2[i,j]
        s = sc[i,j] 
        if ii != 0:
            s -= sc[ii-1,jj-1]
        elif jj != 0:
            raise Exception('dp error.')
        
        btr.append((i, j, s, ii, jj))
        i, j = ii-1, jj-1
    if debug >= 2:
        print 'backtrace:\n' + repr(btr)
    return btr[::-1]


def dp_merge(p1, p2, pth, js1='', js2='', debug=0):
    m = []
    for i, j, s, ii, jj in pth:
        m.append((js1.join(p1[ii:i+1]),js2.join(p2[jj:j+1]),s))
    if debug >= 1:
        print 'result:\n' + repr(m)
    return m

def match_para(ps, f):
    #from tqdm import tqdm
    rps, idx = [], 0
    for p in ps: #tqdm(ps):
        p1, p2 = p[0], p[1]
        path = dp_match(p1, p2, f, debug=0)
        match = dp_merge(p1, p2, path)
        p1 = [s1 for s1, s2, s in match]
        p2 = [s2 for s1, s2, s in match]
        sc = [s for s1, s2, s in match]
        rps.append((p1, p2, sc))
        idx += 1
    return rps

    