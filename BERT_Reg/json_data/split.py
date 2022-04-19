import json
import copy
from tqdm import tqdm
from transformers import BertTokenizer
tz = BertTokenizer.from_pretrained("bert-base-cased")

global_token_threshold = 1200
global_stride = 10

def get_total_tokens(sent_labels):
  # given a list of sentence-like units with score mappings
  # returns an estimate of the total number of tokens in the document

  count = 0
  for unit in sent_labels:
    tokens = tz.tokenize(unit[0])
    count += len(tokens)
  return count

def split(document):
  global global_token_threshold
  global global_stride
  split_collection = []

  lengths = []
  sent_labels = []
  split = []

  for unit in document['sent_labels']:
    token_count = len(tz.tokenize(unit[0]))
    lengths.append(token_count)
    split.append(unit)

    if sum(lengths) < global_token_threshold:
      continue

    else:
      split_collection.append(split)
      stride = min(global_stride, len(split))

      split = split[- stride: ]
      lengths = lengths[- stride: ]
  
  split_collection.append(split)

  split_document = []
  for index, unit in enumerate(split_collection):
    datapoint = {}
    datapoint['id'] = document['id']
#    datapoint['url'] = document['url']
    datapoint['split_index'] = index
    datapoint['sent_labels'] = unit
    split_document.append(datapoint)
  
  return split_document

def post_proc(data):
  # given a processed dataset as input
  # postprocesses each document instance such that 
  # individual datapoints do not exceed global_token_threshold
  # returns the post-processed dataset in the same original format

  processed = []

  for document in tqdm(data):
    token_count = get_total_tokens(document['sent_labels'])
    if token_count < global_token_threshold:
      processed.append(document)
    else:
      split_document = split(document)
      processed += split_document
  
  return processed

# Operating on the val data

data_path = 'infopop/test.json'
with open(data_path, 'r+') as f:
  val_data = json.load(f)

processed_val = post_proc(val_data)

print()
print('Number of datapoints in: ')
print('- raw val dataset: ' + str(len(val_data)))
print('- post-processed val dataset: ' + str(len(processed_val)))

json.dump(processed_val, open('infopop/test1.json', 'w'))
