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

lr = 1e-3
max_grad_norm = 1.0

with open('./data.json') as file:
  dataset = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', max_position_embeddings=MAX_LEN)
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)

def get_padded_input(input):
  if len(input) > MAX_LEN:
    print("Warning: Long sequence input for BERT. Truncating anything larger than {}th token. Actual size: {}".format(MAX_LEN, len(input)))
    input = input[:(MAX_LEN - 1)] + tokenizer.tokenize('[SEP]')
  return pad_sequences([tokenizer.convert_tokens_to_ids(input)], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")[0]

sentence_inputs = []
sentence_labels = []
for item in dataset['sentences_with_questions']:
  if item['question_id'] == QUESTION_ID:
    sentence_inputs.append(tokenizer.tokenize('[CLS] ' + item['question'] + ' [SEP] ' + item['sentence']))
    sentence_labels.append(torch.tensor([item['label']]))
sentence_inputs = [get_padded_input(item) for item in sentence_inputs]
sentence_inputs = [torch.tensor([item]) for item in sentence_inputs]
if n_gpu > 0:
  sentence_inputs = [item.to(device) for item in sentence_inputs]
  sentence_labels = [item.to(device) for item in sentence_labels]
len_sentence = len(sentence_inputs)

article_inputs = []
article_labels = []
for item in dataset['original_data'][QUESTION_ID - 1]:
  article_inputs.append(tokenizer.tokenize('[CLS] ' + item['article'] + ' [SEP]'))
  article_labels.append(torch.tensor([item['answer']]))
article_inputs = [get_padded_input(item) for item in article_inputs]
article_inputs = [torch.tensor([item]) for item in article_inputs]
if n_gpu > 0:
  article_inputs = [item.to(device) for item in article_inputs]
  article_labels = [item.to(device) for item in article_labels]
len_article = len(article_inputs)

epochs = 15

for _ in trange(epochs, desc="Epoch"):
  count_sentence = 0
  count_article = 0
  running_loss = 0.0
  for i in range(len_sentence + len_article):
    train_sentence = count_sentence < len_sentence and \
                     (count_article * (len_sentence / len_article) >= count_sentence)
    input = sentence_inputs[count_sentence] if train_sentence else article_inputs[count_article]
    label = sentence_labels[count_sentence] if train_sentence else article_labels[count_article]
    output = model(input)
    loss = criterion(output[0], label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    running_loss += loss.item()
    if train_sentence:
      count_sentence += 1
    else:
      count_article += 1
    if i % 200 == 199:    # print every 200 mini-batches
      print('[%5d] loss: %.3f' % (i + 1, running_loss / 200))
      running_loss = 0.0

Path('./bert_trained').mkdir(parents=True, exist_ok=True)
model.save_pretrained('./bert_trained')
