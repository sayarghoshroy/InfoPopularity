# Processing the CNN Dailymail Dataset
# To compute sentence salience scores
# Based on ROUGE overlaps between sentence and Gold Summary

# Importing packages
from datasets import load_dataset
from tqdm import tqdm
from rouge import Rouge
import numpy as np
import json
import nltk
import time
import os.path
from os import path

nltk.download('punkt')

dataset = load_dataset('cnn_dailymail', '3.0.0')
print('Dataset loaded.', flush = True)

# print('Data Format...')
# print(dataset)

def extract_point(unit_num):
  global dataset
  try:
    unit_num = int(unit_num)
    blob = dataset['train']
    result = [blob['id'][unit_num], blob['highlights'][unit_num], blob['article'][unit_num]]
    return result
  
  except Exception as E:
    return -1

# Verifying Load Correctness
# dataset['train']['highlights'][0]

rouge = Rouge()

def clean(unit):
  unit = unit.strip()
  unwanted = ['(CNN)']
  for blob in unwanted: 
    unit = unit.replace(blob, '') 

  unit.replace('\n', ' ')
  return unit

def get_token_string(unit):
  tokens = nltk.word_tokenize(unit)
  token_string = ''
  for token in tokens:
    token_string = token_string + token + ' '
  return token_string.strip()

def get_score_mappings(article, highlights):
  suppress_logs = 1

  split_sentences = nltk.sent_tokenize(clean(article))
  sentences = [get_token_string(sentence) for sentence in split_sentences]
  highlights = get_token_string(clean(highlights))

  # To check pre-processing steps
  if suppress_logs != 1:
    print('Article Sentences: ' + str(sentences))
    print('Article Highlight: ' + str(highlights))
  
  score_set = []
  for sentence in sentences:
    try:
      scores = rouge.get_scores(sentence.lower(), highlights.lower())[0]
      unit_score = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
    except Exception as e:
      if suppress_logs != 1:
        print('ROUGE Scoring Error')
        print('Sentence: ' + str(sentence))
        print('Highlight: ' + str(highlights))
      unit_score = [0, 0, 0]
    
    score_set.append(unit_score)
    
  score_set_size = len(score_set)
  sum_of_scores = np.ndarray.tolist(np.sum(np.asarray(score_set), axis = 0))
  
  for metric in range(3):
    # For ROUGE 1, 2, and L
    factor = sum_of_scores[metric]
    if factor > 0:
      # Else, score_set[index][metric] is 0 as well
      for index in range(score_set_size):
        score_set[index][metric] /= factor
  
  mappings = []
  for index in range(score_set_size):
    box = [sentences[index]]
    for metric in range(3):
      box.append(score_set[index][metric])
    mappings.append(box)
  
  return mappings

test_mode = 0
test_mode_limit = 100
interval = 500

for set_type in ['train', 'validation', 'test']:
  print('Building the ' + str(set_type) + ' set.', flush = True)
  print('', flush = True)

  work_on = dataset[set_type]
  set_size = len(work_on['id'])

  use_size = set_size
  if test_mode == 1:
    use_size = test_mode_limit

  print('Building the ' + set_type + ' split.', flush = True)
  print('Number of Datapoints: ' + str(set_size), flush = True)
  print('', flush = True)

  data_split = []

  backup_present = path.isfile(set_type + '.json')
  done = 0

  if backup_present:
    print('Pre-processed Data Backup Found: ' + str(backup_present), flush = True)
    with open(set_type + '.json', 'r+') as f:
      data_split = json.load(f)
    done = len(data_split)
    print('Starting from ' + str(done) + ' onwards.', flush = True)

  for index in tqdm(range(done, use_size)):
    article = work_on['article'][index]
    highlights = work_on['highlights'][index]
    id = work_on['id'][index]

    mappings = get_score_mappings(article, highlights)
    data_point = {}
    data_point['id'] = id
    data_point['sent_labels'] = mappings

    data_split.append(data_point)

    if index % interval == 0:
      with open(set_type + '.json', 'w+') as f:
        json.dump(data_split, f)

    if test_mode == 1 and index > test_mode_limit:
      break

  print('', flush = True)
  time.sleep(1)
  with open(set_type + '.json', 'w+') as f:
    json.dump(data_split, f)

# Testing out a sample unit creation
# unit = extract_point(1657)
# get_score_mappings(unit[2], unit[1])

# ^_^ Thank You