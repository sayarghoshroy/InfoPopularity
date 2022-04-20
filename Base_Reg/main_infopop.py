#!/usr/bin/env python3

import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from time import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='/home/userName/Summa_data/InfoPop/train.json')
parser.add_argument('-val_dir',type=str,default='/home/userName/Summa_data/InfoPop/val.json')
parser.add_argument('-embedding',type=str,default='embedding_glove_100d.npz')
parser.add_argument('-word2id',type=str,default='word2id_glove_100d.json')
parser.add_argument('-report_every',type=int,default=1500)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-foldername' ,type=str,default='def_folder')

parser.add_argument('-topk',type=int,default=15)
parser.add_argument('-startind',type=int,default=0)
parser.add_argument('-endind',type=int,default=15)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

def eval(net,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        features,targets,_,doc_lens = vocab.make_features(batch)
        features,targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features,doc_lens)
        loss = criterion(probs,targets)
        total_loss += loss.item()
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    print("loss" , loss)
    return loss

def predict():
    embed = torch.Tensor(np.load(args.embedding)['embedding'])

    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    ind = 0
    
    with open(args.filename) as f:
        data = json.loads(f.read())
        print(len(data))

        pred_dataset = utils.Dataset(data)

        pred_iter = DataLoader(dataset=pred_dataset,batch_size=args.batch_size,shuffle=False)
        doc_num = len(pred_dataset)
        ind = 0
        time_cost = 0
        file_id = 1

        for batch in tqdm(pred_iter):
            if ind < args.startind or ind >= args.endind : 
                ind+=1
                continue
            features, doc_lens = vocab.make_predict_features(batch)
            if (doc_lens[0] < 10) : 
                ind+=1
                continue 
            t1 = time()
            if use_gpu or False:
               probs = net(Variable(features).cuda(), doc_lens)
            else:
                probs = net(Variable(features), doc_lens)
            t2 = time()
            time_cost += t2 - t1
            start = 0
            for doc_id,doc_len in enumerate(doc_lens):
                stop = start + doc_len
                prob = probs[start:stop]

                for i , p in enumerate(prob):
                    data[ind]['sent_labels'][i].append(p.item())

            ind+=1

    data = data[args.startind:args.endind]
    with open( str(args.startind) + "-" + str(args.endind) +  "predicted.json" , "w+") as f:
        json.dump(data ,f , indent = 4,ensure_ascii=False)

if __name__=='__main__':
    if args.test:
        test()
    elif args.predict:
        predict()
    else:
        train()