
# coding: utf-8

# In[ ]:


import json
import os
import pickle
import torch
from pathlib import Path
from typing import Iterable
from tqdm import tqdm


from utils import Embedding

from torch.utils.data import DataLoader



from dataset import *


# In[ ]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


# In[ ]:


# Get Training data (after preprocess)

file = open('extractive_train.pkl', 'rb')
train = pickle.load(file)
file.close()


# In[ ]:


# Create dataloader

dataloader = DataLoader(
    dataset = train,
    batch_size = 128,
    shuffle = True,
    collate_fn = lambda x: SeqTaggingDataset.collate_fn(train, x)
)


# In[ ]:


#Create Model

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
    


lstm = LSTM().cuda()


# In[ ]:


#Setting optimizer & Loss function
import torch.optim as optim
pos_weight = torch.tensor(6.84)
loss_function = nn.BCEWithLogitsLoss(pos_weight = pos_weight, reduction='none').cuda()
optimizer = optim.Adam(lstm.parameters(), lr=0.00001)


# In[ ]:


# Train model

for epoch in range(5):
    total_loss = []
    for i, data in enumerate(dataloader):
        
        data_x, data_y = data['text'], data['label']
        length = [(300 if sr[-1][1]>300 else sr[-1][1])for sr in data['sent_range']]
    
        optimizer.zero_grad()
    
        output = lstm(data_x.cuda(), length)
    
        data_y = data_y.view(data_y.size()[0],data_y.size()[1], 1)
        data_y = data_y.cuda()
        batch_size = data_y.size()[0]
        
        mask = torch.zeros(batch_size,300,1).cuda()
        mask = mask.ge(1)
        for j,ln in enumerate(length,0):
            for k in range(300):
                if k >= ln:
                    break
                mask[j][k][0] = True
        losses = torch.masked_select(loss_function(output, (data_y.float())) , mask)
        mask = 0
        total_loss.append(losses.mean())
        losses.mean().backward()
        optimizer.step()
        
    #avg loss each epoch    
    print(sum(total_loss)/len(total_loss))
        
        




# In[ ]:


torch.save(lstm.state_dict(), 'extractive_lstm.pkl') 

