import torch
from transformers import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import trange
import numpy as np
from pathlib import Path

import json
import sys

pretrained = './bert_trained'

np.random.seed(2020)

MAX_LEN = 512
QUESTION_ID = 1

with open('./data.json') as file:
  dataset = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(pretrained, max_position_embeddings=MAX_LEN)

def get_padded_input(input):
  if len(input) > MAX_LEN:
    input = input[:(MAX_LEN - 1)] + tokenizer.tokenize('[SEP]')
  return pad_sequences([tokenizer.convert_tokens_to_ids(input)],
                       maxlen=MAX_LEN,
                       dtype="long",
                       truncating="post",
                       padding="post")[0]

def downsample_inputs(inputs, labels):
  neg_count = 0
  pos_count = 0
  for label in labels:
    if label.item() == 1:
      pos_count += 1
    else:
      neg_count += 1
  if pos_count == neg_count:
    return inputs, labels

  if pos_count > neg_count:
    downsampled_label = 1
    downsample_rate = neg_count / pos_count
  else:
    downsampled_label = 0
    downsample_rate = pos_count / neg_count
  new_inputs = []
  new_labels = []
  for input, label in zip(inputs, labels):
    if label.item() != downsampled_label or np.random.random_sample() < downsample_rate:
      new_inputs.append(input)
      new_labels.append(label)
  return new_inputs, new_labels

sentence_inputs = []
sentence_labels = []
for item in dataset['sentences_with_questions']:
  if item['question_id'] == QUESTION_ID:
    sentence_inputs.append(tokenizer.tokenize('[CLS] ' + item['question'] + ' [SEP] ' + item['sentence']))
    sentence_labels.append(torch.tensor([1 if item['label'] > 0 else 0]))
sentence_inputs = [get_padded_input(item) for item in sentence_inputs]
sentence_inputs = [torch.tensor([item]) for item in sentence_inputs]
if n_gpu > 0:
  sentence_inputs = [item.to(device) for item in sentence_inputs]
  sentence_labels = [item.to(device) for item in sentence_labels]
  model.to(device)
sentence_inputs, sentence_labels = downsample_inputs(sentence_inputs, sentence_labels)

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

for input, label in zip(sentence_inputs, sentence_labels):
  output = model(input)
  output = output[0].detach().numpy()[0]
  output = 0 if output[0] > output[1] else 1
  if output == 0 and label == 0:
    true_negative += 1
  elif output == 0 and label == 1:
    false_negative += 1
  elif output == 1 and label == 0:
    false_positive += 1
  elif output == 1 and label == 1:
    true_positive += 1

print('True positive: ', true_positive)
print('False positive: ', false_positive)
print('True negative: ', true_negative)
print('False negative: ', false_negative)
