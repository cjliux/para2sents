# -*- coding: utf-8 -*-
import os
import string
import torch as th
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from utils import read_mono

data_dir = os.path.join(os.path.split(__file__)[0], '../real_data/test')

"""
w2v
"""
def train_w2v(corpus, n_x=100, ws=5, mc=5, wkr=4):
    ss = read_mono(corpus)
    ss = [ string.lower(s).split() for s in ss ]
    md = Word2Vec(ss, size=n_x, window=ws, min_count=mc, workers=wkr)
    return md

def save_w2v(data_dir, chm, enm):
    chm.wv.save_word2vec_format(
            os.path.join(data_dir,'w2v','ch_w2v.txt'),
            os.path.join(data_dir,'w2v','ch_w2v.voc'))
    enm.wv.save_word2vec_format(
            os.path.join(data_dir,'w2v','en_w2v.txt'),
            os.path.join(data_dir,'w2v','en_w2v.voc'))

def load_w2v(data_dir):
    chm = KeyedVectors.load_word2vec_format(
            os.path.join(data_dir,'w2v','ch_w2v.txt'))
    enm = KeyedVectors.load_word2vec_format(
            os.path.join(data_dir,'w2v','en_w2v.txt'))
    return chm, enm


"""
bli
"""
class Dict(Dataset):
    # TODO
    def __init__(self, fdict, chm, enm):
        self.len = 0
        self.chm = chm 
        self.enm = enm
        with open(fdict, 'rt') as fd: 
            self.wp = [x.strip().decode('utf-8').split() for x in fd if x.strip()]
            self.wp = [p for p in self.wp if p[0] in chm.wv.vocab.keys()
                            and p[1] in enm.wv.vocab.keys()]
            self.len = len(self.wp)
            
    def __getitem__(self,idx):
        cw = self.chm.wv.vocab[self.wp[idx][0]].index
        ce = self.chm.wv[self.wp[idx][0]]
        ew = self.enm.wv.vocab[self.wp[idx][1]].index
        ee = self.enm.wv[self.wp[idx][1]]
        return cw, ce, ew, ee
    
    def __len__(self):
        return self.len

class Model(th.nn.Module):
    def __init__(self, ces, ees):
        super(Model,self).__init__()
        self.lin = th.nn.Linear(ces, ees)
        
    def forward(self, ce):
        pee = self.lin(ce)
        return pee

def save_bli(bli, mdt, e=None):
    if not os.path.exists(mdt):
        os.mkdir(mdt)
    fname = "bli.pt"
    if e is not None:
        fname = "bli_e{}.pt".format(e)
    th.save(bli, os.path.join(mdt, fname))

def load_bli(mdf, fname=None):
    if fname is None: 
        fname="bli.pt"
    return th.load(os.path.join(mdf, fname))
    
def train_bli(chm, enm, fdict, mdt=None):
    print 'load data'
    loader = DataLoader(Dict(fdict, chm, enm),
         2, shuffle=True, num_workers=3)
    
    print 'build model'
    bli = Model(chm.wv.vector_size, enm.wv.vector_size).cuda()
    crit = th.nn.MSELoss(size_average=False).cuda()
    opt = th.optim.SGD(bli.parameters(), lr=0.0015, momentum=0.5)

    print 'train model'
    bli.train()
    max_e = 5000
    for e in range(max_e):
        for b, (cw, cwe, ew, ewe) in enumerate(loader):    
            cwe = Variable(cwe, requires_grad=False).cuda()
            ewe = Variable(ewe, requires_grad=False).cuda()
            
            pewe = bli(cwe)
            loss = crit(pewe, ewe)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if (b+1) % (len(loader)/5) == 0:
                print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        e,(b+1) * len(cw), len(loader.dataset), 
                        100. * b / len(loader), loss.data[0])
        if (e+1) % (max_e/10) == 0 and mdt is not None:
            save_bli(bli, mdt, e)
            print 'model saved @ epoch {}'.format(e)
           
    if mdt is not None:
        save_bli(bli, mdt)  
        print 'final model saved.'
    
    return bli

def test_bli(bli, chm, enm, fdict):
    print 'load data'
    loader = DataLoader(Dict(fdict, chm, enm),
            24, shuffle=True, num_workers=3)
    ee = Variable(th.Tensor(enm.syn0), requires_grad=False).cuda()
    
    bli.eval()
    loss = 0
    correct = 0
    for cw, cwe, ew, ewe in tqdm(loader):
        cwe = Variable(cwe, requires_grad=False).cuda()
        ew = Variable(ew, requires_grad=False).cuda()
        ewe = Variable(ewe, requires_grad=False).cuda()
        
        pewe = bli(cwe)
        # sum up batch loss
        loss += F.mse_loss(pewe, ewe, size_average=False).data[0]
        
        pred = th.matmul(pewe,th.t(ee)).data.max(1, keepdim=True)[1]
        correct += pred.eq(ew.data.view_as(pred)).cpu().sum()

    loss /= len(loader.dataset)
    #print '\nTest set: Average loss: {:.4f}\n'.format(loss)
    print '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset))


def main(data_dir, ch_corpus, en_corpus, fdict, mdt):
    #chm, enm = train_w2v(ch_corpus), train_w2v(en_corpus)
    #save_w2v(data_dir, chm, enm)
    chm, enm = load_w2v(data_dir)
    
    #bli = train_bli(chm, enm, fdict, mdt)
    bli = load_bli(mdt)
    test_bli(bli, chm, enm, fdict)
    
    
if __name__=='__main__':
    corpus = ['ch_seg', 'en_seg']
    corpus = [os.path.join(data_dir, x) for x in corpus]
    fdict = os.path.join(data_dir, "dict/fdict")
    mdt = os.path.join(data_dir, "bli")
    main(data_dir, corpus[0], corpus[1], fdict, mdt)
