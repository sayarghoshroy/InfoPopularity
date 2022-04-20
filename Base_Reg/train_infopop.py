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
parser.add_argument('-lr',type=float,default=1e-5)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='../CNN_DailyMail/train.json')
parser.add_argument('-val_dir',type=str,default='../CNN_DailyMail/val.json')
parser.add_argument('-embedding',type=str,default='embedding_glove_100d.npz')
parser.add_argument('-word2id',type=str,default='word2id_glove_100d.json')
parser.add_argument('-report_every',type=int,default=200)
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
    
def eval(net,vocab,data_iter,criterion):
    print("IN EVAL")
    net.eval()
    total_loss = 0
    batch_num = 0
    print(len(data_iter))
    with torch.no_grad():
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

g_lr = args.lr

def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')
    
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    import math
    def process_doc(ex):
        ret = {}
        labels = []
        doc = []
        for sent in ex:
            doc.append(sent[0])
            labels.append(str(float(sent[1]*1)))
            if math.isnan(float(labels[-1])):
                labels[-1] = "0"
        ret['doc'] = "\n".join(doc)
        ret['labels'] = "\n".join(labels)
        #print(ret['labels'])
        return ret

    with open(args.val_dir) as f:
        examples = [process_doc(ex['sent_labels']) for ex in json.loads(f.read())][:]
    val_dataset = utils.Dataset(examples)

    with open(args.train_dir) as f:
        examples = [process_doc(ex['sent_labels']) for ex in json.loads(f.read())][:]
    train_dataset = utils.Dataset(examples)

    print("Length of train " , len(examples))
    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models,args.model)(args,embed)
    if use_gpu:
        net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True)

    val_iter = DataLoader(dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False)
    # loss function
    criterion = nn.MSELoss()
    # model info
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    min_loss = float('inf')
    global g_lr
    t1 = time() 
    loss_arr = []
    for epoch in range(1,args.epochs+1):
        flag = False
        optimizer = torch.optim.Adam(net.parameters(),lr = g_lr)
        net.train()
        for i,batch in enumerate(train_iter):
            features,targets,_,doc_lens = vocab.make_features(batch)
            features,targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features,doc_lens)

            loss = criterion(probs,targets)

            print(loss)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if args.debug:
                print('Batch ID:%d Loss:%f' %(i,loss.data[0]))
                continue

            if i % args.report_every == 0:
                cur_loss = eval(net,vocab,val_iter,criterion)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_path = net.save()
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                        % (epoch,min_loss,cur_loss))
        if len(loss_arr):
            if np.abs(cur_loss - loss_arr[-1]) < 1e-8 :
                flag = True
                break
            if cur_loss > loss_arr[-1] : g_lr/=1.5 
        loss_arr.append(cur_loss)
        
    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))

def predict():
    pass 

if __name__=='__main__':
    if args.test:
        test()
    elif args.predict:
        predict()
    else:
        train()