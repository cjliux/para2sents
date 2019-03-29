# -*- coding: utf-8 -*-
import re
#import uniout
import numpy as np
from collections import Counter

"""
mono
"""
def read_mono(fname):
    with open(fname, 'r', encoding='utf8') as fd:
        lns = [s for s in fd]
        return [s.strip() for s in lns if len(s.strip()) > 0]

def save_mono(fname, ss):
    with open(fname, 'w', encoding='utf8') as fd:
        for s in ss:
            fd.write(s + '\n')

"""
paras
"""
def read_para(fname1, fname2=None, revr=False):
    #print fname1, fname2, revr
    if fname2 is None:
        with open(fname1, 'r', encoding='utf8') as fd:
            lines = fd.readlines()
        ps = []
        idx = 0
        while idx < len(lines):
            buf = []
            while idx < len(lines) and len(lines[idx].strip()) == 0:
                idx += 1
            if idx >= len(lines): break
            while idx < len(lines) and len(lines[idx].strip()) != 0:
                buf.append(lines[idx].strip())
                idx += 1
            if not revr:
                ps.append((buf[0], buf[1]))    
            else:
                ps.append((buf[1], buf[0]))
        return ps
    else:
        if not revr:
            return zip(read_mono(fname1),read_mono(fname2))
        else:
            return zip(read_mono(fname2),read_mono(fname1))
        
def save_para(ps, fname1, fname2=None, revr=False, sep=False):
    if fname2 is None:
        with open(fname1, 'w', encoding='utf8') as fd:
            for p in ps:
                if not revr:
                    fd.write(p[0] + '\n')
                    if sep: fd.write('\n')
                    fd.write(p[1] + '\n\n')
                else:
                    fd.write(p[1] + '\n')
                    if sep: fd.write('\n')
                    fd.write(p[0] + '\n\n')
    else:
        with open(fname1, 'w', encoding='utf8') as fd1:
            with open(fname2, 'w', encoding='utf8') as fd2:
                for p in ps:
                    if not revr:
                        fd1.write(p[0] + '\n')
                        if sep: fd.write('\n')
                        fd2.write(p[1] + '\n')
                        if sep: fd.write('\n')
                    else:
                        fd1.write(p[1] + '\n')
                        if sep: fd.write('\n')
                        fd2.write(p[0] + '\n')
                        if sep: fd.write('\n')

def split_para(ps, sre1=u'(。)', sre2=u'(\.)', eat=False, debug=0):
    nps = []
    for p in ps:
        p1,p2,np1,np2,s = p[0],p[1],[],[],0
        for it in re.finditer(sre1, p1):
            if debug >= 1:
                print(it.start(), it.end(), it.group(1))
            e = it.start() if eat else it.end()
            np1.append(p1[s:e])
            s = it.end()
        np1.append(p1[s:]) if s < len(p1) else None
        s = 0
        for it in re.finditer(sre2, p2):
            if debug >= 1:
                print(it.start(), it.end(), it.group(1))
            e = it.start() if eat else it.end()
            np2.append(p2[s:e])
            s = it.end()
        np2.append(p2[s:]) if s < len(p2) else None
        nps.append((np1, np2) + p[2:])
        if debug:
            break
    return nps

def join_para(ps, con1='\t', con2='\t'):
    nps = []
    for p in ps:
        t = (con1.join(p[0]), con2.join(p[1]))
        nps.append(t + p[2:])
    return nps


def split_real(p, eat=False):
    sep = np.zeros((len(p),)).astype(np.int)
    nbr = np.zeros((len(p),)).astype(np.int)
    # html tag
    for it in re.finditer(u"<\s*(?P<tag>[^\s>]+)[^>]*/\s*>",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"<\s*(?P<tag>[^\s>]+)[^>]*>(.|\n)*?<\s*/(?P=tag)\s*>",p):
        nbr[it.start():it.end()] = 1
    
    # url
    ipv4 = u"(((\d{1,2})|(1\d{2})|(2[0-4]\d)|(25[0-5]))\.){3}((\d{1,2})|(1\d{2})|(2[0-4]\d)|(25[0-5]))"
    for it in re.finditer(ipv4,p):
        nbr[it.start():it.end()] = 1
    fsfx = u"\.(zip|rar|html|htm|php|aspx|jsp|txt|json|js)"
    for it in re.finditer(fsfx,p):
        nbr[it.start():it.end()] = 1
    dnm = u"([\w]+(-[\w]+)\s*\.\s*)*[\w]+(-[\w]+)\s*\.\s*(com|net|org)"
    for it in re.finditer(dnm, p):
        nbr[it.start():it.end()] = 1
    #dnm = u"[\w]+(-[\w]+)*(\s*\.\s*[\w]+(-[\w]+)*)+"
    #path = u"(/\s*[\w-~+&@#%=|](\s*\.\s*[\w-~+&@#%=|])*(\s*/)?"
    #qps = u"[\w-]\s*=(\s*[\w-~+@#%|])?(\s*&\s*[\w-]\s*=(\s*[\w-~+@#%|])?)*"
    #url = u"(https?|ftp|file)\s*:\s*/\s*/\s*("+ipv4+"|"+dnm+")\s*:\s*\d+\s*" + path + "(\s*?\s*"+qps+")?"
    
    # numerical
    for it in re.finditer(u"\d+\s*\.\s*\d+\s*\.\s*\d+|(\$|￥)?\d*\.\d+(%)?|\d{1,2}\s*°\s*\d{1,2}'(\d{1,2}(.\d{1,2})?\")?",p):
        nbr[it.start():it.end()] = 1
    
    # gramma
    for it in re.finditer(u"['’]\s*(t|m|s|re|ve|d|em|er|am|ll)\s+",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"\w+s'\s+",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"^\s*\d+\s*\.",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"(Mr|Mrs|Miss|Dr|No|Co)\s*\.",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"([A-Z]|Jr)\s*\.",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"(Jan|Feb|Mar|Apr|May|June|July|Aug|Sept|Oct|Nov|Dec|Mon|Tues|Wed|Thur|Fri|Sat|Sun)\s*\.",p):
        nbr[it.start():it.end()] = 1
    for it in re.finditer(u"(p|a)\s*\.\s*m\s*\.|i\s*\.\s*e\s*\.",p):
        nbr[it.start():it.end()] = 1
        
    # special
    for it in re.finditer(u"\bE\s*!\b|\bBin\s*.\s*E\b", p): 
        nbr[it.start():it.end()] = 1
    
    # open-close pairs
    def recur_lvs(p, l, r):
        lvs1 = np.zeros((len(p),)).astype(np.int)
        for i,c in enumerate(p):
            if nbr[i] == 0:
                if c in l:
                    lvs1[i] = 1 if i == 0 else lvs1[i-1]+1
                elif c in r:
                    lvs1[i] = max(lvs1[i-1]-1, 0)
                else:
                    lvs1[i] = 0 if i == 0 else lvs1[i-1]
            else:
                lvs1[i] = 0 if i == 0 else lvs1[i-1]
        lvs2 = np.zeros((len(p),)).astype(np.int)
        for i,c in enumerate(reversed(p)):
            if nbr[-(i+1)] == 0:
                if c in r:
                    lvs2[i] = 1 if i == 0 else lvs2[i-1]+1
                elif c in l:
                    lvs2[i] = max(lvs2[i-1]-1, 0)
                else:
                    lvs2[i] = 0 if i == 0 else lvs2[i-1]
            else:
                lvs2[i] = 0 if i == 0 else lvs2[i-1]
        lvs = np.maximum(lvs1, lvs2[::-1])
        lvs[:-1] = np.minimum(lvs[:-1],lvs[1:])
        if p[-1] in r:
            lvs[-1] -= 1
        return lvs
    
    for i, (l,r) in enumerate(
            zip([u"“‘", u"（《({["], [u"”’", u"）》)}]"])):
        lvs = recur_lvs(p, l, r)
        nbr[lvs!=0] = 1
    
    # matched pairs
    def mark_odr(p, m):
        odr = np.zeros((len(p),)).astype(np.int)
        for i,c in enumerate(p):
            if nbr[i] == 0 and c in m:
                odr[i] = 1 if i == 0 else odr[i-1]+1
            else:
                odr[i] = 0 if i == 0 else odr[i-1]
        return odr
    
    for m in [u"\"", u"'"]:
        odr = mark_odr(p, m)
        for l in range(1,np.amax(odr)+1,2):
            nbr[odr==l] = 1
        
    # ref proc
    x = 0
    while x < len(nbr):
        s = x
        while s < len(nbr) and nbr[s] == 0: s+=1
        if s >= len(nbr): break   # finished
        e = s
        while e < len(nbr) and nbr[e] != 0: e+=1
        x = e
        
        if e >= len(nbr) or p[e] not in u"”’\"'": 
            continue  # unclosed
        else:
            ref = p[s:e].strip()
            if len(ref) > 0 and ref[-1] in u"。？！.?!":
                sep[e] = 1 
        
    # common sep
    for it in re.finditer(u'[。！？.!?][\s。！？.!?]*', p):
        if np.sum(nbr[it.start():it.end()]) > 0:
            continue
        sep[it.start():it.end()] = 1
    
    # split
    i = 0
    pp = []
    while i < len(p):
        s = i
        while s < len(p) and sep[s] != 1: s+=1
        e = s
        while e < len(p) and sep[e] != 0: e+=1
        if eat: pp.append(p[i:s].strip())
        else: pp.append(p[i:e].strip())
        i = e
    
    return pp


"""
lang
"""
def is_zh(s):
    m = re.match(u'[\u4e00-\u9fa5]+',s)
    if m is not None and m.span() == (0, len(s)):
        return True
    else:
        return False
    
"""
func
"""
def IoU(x, y):
    x, y = Counter(x), Counter(y)
    i = sum((x-(x-y)).values())
    u = sum((x+(y-x)).values())
    return float(i) / u

def fbeta(x, y, b=1):
    x, y = Counter(x), Counter(y)
    p = 1 - float(sum((y-x).values())) / sum(y.values())
    r = 1 - float(sum((x-y).values())) / sum(x.values())
    nu, de = (1.+b**2)*p*r, p*b**2+r
    if nu == 0: return 0
    else: return nu / de
    
def align(p1, p2, f=fbeta):
    return np.asarray([[f(x,y) for y in p2] for x in p1])

