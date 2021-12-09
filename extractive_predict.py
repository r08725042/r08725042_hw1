
# coding: utf-8

# In[ ]:


from torch.utils.data import DataLoader
import os
import json
import pickle
import torch
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from torch.utils.data import Dataset
from dataset import SeqTaggingDataset
from utils import Tokenizer, Embedding, pad_to_len
from argparse import ArgumentParser


# In[ ]:


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()


# In[ ]:


test_path = Path(args.test_data_path)

with open(test_path) as f:
    test = [json.loads(line) for line in f]


# In[ ]:


tokenizer = Tokenizer(lower=True)
print('loading embedding...')
with open('embedding.pkl', 'rb') as f:
        embedding = pickle.load(f)
        
tokenizer.set_vocab(embedding.vocab)


# In[ ]:


from preprocess_seq_tag import process_seq_tag_samples, get_tokens_range, create_seq_tag_dataset 

print('data preprocessing...')
create_seq_tag_dataset(
        process_seq_tag_samples(tokenizer, test),
        'extractive_test.pkl',
        tokenizer.pad_token_id
    )


# In[ ]:


file = open('extractive_test.pkl','rb')    
test = pickle.load(file)      
file.close()

testloader = DataLoader(
    dataset = test,
    batch_size = 32,
    shuffle = False,
    collate_fn = lambda x: SeqTaggingDataset.collate_fn(test, x)
)


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors.cuda()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.lstm = nn.LSTM(300,512,3, bidirectional=True, batch_first = True)
        self.hidden2hid = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.5)

   
        
        
    def forward(self, inputs, length):


        embed = self.dropout(self.embedding(inputs))
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, length, batch_first=True,enforce_sorted=False)
        lstm_out, hidden = self.lstm(embed)
        tag, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        tag_space = self.hidden2hid(tag)


        return tag_space
    

print('load model weight...')
lstm = LSTM().cuda()
lstm.load_state_dict(torch.load('extractive_lstm.pkl'))


# In[ ]:

print('now predicting...')

y_p = []
y_id = []
for i, y in enumerate(testloader):
    data_x = y['text']
    for ids in y['id']:
        y_id.append(ids)
    length = [(300 if sr[-1][1]>300 else sr[-1][1])for sr in y['sent_range']]
    lstm.eval()
    output = lstm(data_x.cuda(), length)
    
    for j,label in enumerate(output,0):
        summ_point = []
        for k in y['sent_range'][j]:
            tempv = label[k[0]:k[1]]
            temp_point = [l[0] for l in tempv]
            summ_point.append(sum(temp_point))
        temp_max =[]
        max_index = summ_point.index(max(summ_point))
        temp_max.append(max_index)
        summ_point[max_index] = -100
        max_index = summ_point.index(max(summ_point))
        temp_max.append(max_index)


        y_p.append(temp_max)


# In[ ]:

print('create predict file')
data = []
for i in range(len(y_p)):

    temp = {'id':y_id[i], 'predict_sentence_index' : y_p[i]}
    data.append(temp)


# In[ ]:


output_file = Path(args.output_path)
output_file.write_text(
    '\n'.join(
        [
            json.dumps(line) for line in data
        ]
    ) + '\n'
)

