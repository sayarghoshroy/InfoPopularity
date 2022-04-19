import json
import math
from tqdm import tqdm
from nltk.tokenize import word_tokenize

f = open('cnn/test.json', 'r')
f1 = json.load(f)
l = len(f1)
final = []
for i in tqdm(f1):
        d = {}
        l1 = []
        l2 = []
        x = i['sent_labels']
        flag = 0
        for j in x:
            if math.isnan(float(j[1])):
                flag = 1
            l1.append(word_tokenize(j[0]))
            l2.append(j[1])
        d['src'] = l1
        d['label'] = l2
        if flag == 0:
            final.append(d)
        else:
            print('Check!')
    
f2  = '../../textrank/cnn.json'
f2 = open(f2, 'w')
json.dump(final, f2)

