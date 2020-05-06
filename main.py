import torch
from transformers import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import trange
import numpy as np
from pathlib import Path

import json
import sys

np.random.seed(2020)

MAX_LEN = 512
QUESTION_ID = 1

batch_size = 100
lr = 1e-3
max_grad_norm = 1.0

with open('./train_data.json') as file:
  dataset = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', max_position_embeddings=MAX_LEN)
criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.CrossEntropyLoss()
optimizer1 = AdamW(model.parameters(), lr=lr, correct_bias=False)
optimizer2 = AdamW(model.parameters(), lr=lr, correct_bias=False)

def get_padded_input(input):
  if len(input) > MAX_LEN:
    print("Warning: Long sequence input for BERT. Truncating anything larger than {}th token. Actual size: {}".format(MAX_LEN, len(input)))
    input = input[:(MAX_LEN - 1)] + tokenizer.tokenize('[SEP]')
  return pad_sequences([tokenizer.convert_tokens_to_ids(input)], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")[0]

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

  print(pos_count, neg_count, downsampled_label, downsample_rate)
  new_inputs = []
  new_labels = []
  for input, label in zip(inputs, labels):
    if label.item() != downsampled_label or np.random.random_sample() < downsample_rate:
      new_inputs.append(input)
      new_labels.append(label)

  print(len(new_inputs))
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
  criterion1.to(device)
  criterion2.to(device)
sentence_inputs, sentence_labels = downsample_inputs(sentence_inputs, sentence_labels)
len_sentence = len(sentence_inputs)

article_inputs = []
article_labels = []
for item in dataset['original_data'][QUESTION_ID - 1]:
  article_inputs.append(tokenizer.tokenize('[CLS] ' + item['article'] + ' [SEP]'))
  article_labels.append(torch.tensor([1 if item['answer'] == 1 else 0]))
article_inputs = [get_padded_input(item) for item in article_inputs]
article_inputs = [torch.tensor([item]) for item in article_inputs]
if n_gpu > 0:
  article_inputs = [item.to(device) for item in article_inputs]
  article_labels = [item.to(device) for item in article_labels]
article_inputs, article_labels = downsample_inputs(article_inputs, article_labels)
len_article = len(article_inputs)

epochs = 100

for _ in trange(epochs, desc="Epoch"):
  count_sentence = 0
  count_article = 0
  running_loss = 0.0
  for i in range((len_sentence + len_article) // batch_size):
    train_sentence = count_sentence < len_sentence and \
                     (count_article * (len_sentence / len_article) >= count_sentence)
    inputs = sentence_inputs[count_sentence : count_sentence + batch_size] if train_sentence \
            else article_inputs[count_article : count_article + batch_size]
    labels = sentence_labels[count_sentence : count_sentence + batch_size] if train_sentence \
            else article_labels[count_article : count_article + batch_size]
    for input, label in zip(inputs, labels):
      outputs = model(input)
      loss = criterion1(outputs[0], label) if train_sentence else criterion2(outputs[0], label)
      loss.backward()
    
    op = optimizer1 if train_sentence else optimizer2
    op.step()
    op.zero_grad()
    running_loss += loss.item()
    if train_sentence:
      count_sentence += batch_size
    else:
      count_article += batch_size
    if i % 5 == 4:    # print every 5 mini-batches
      print('[%5d] loss: %.3f' % (i + 1, running_loss / 200))
      running_loss = 0.0

Path('./bert_trained').mkdir(parents=True, exist_ok=True)
model.save_pretrained('./bert_trained')
