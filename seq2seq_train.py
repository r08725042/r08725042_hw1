
# coding: utf-8

# In[ ]:


import json
import os
import pickle
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Iterable
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from dataset import *


# In[ ]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


# In[ ]:


# Get Training data (after preprocess)

file = open('abstractive_train.pkl', 'rb')
train = pickle.load(file)
file.close()

# Create dataloader

dataloader = DataLoader(
    dataset = train,
    batch_size = 32,
    shuffle = True,
    collate_fn = lambda x: Seq2SeqDataset.collate_fn(train, x)
)


# In[ ]:


#Create Model

import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors.cuda()
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.gru = nn.GRU(300,512, bidirectional=True, batch_first = True)
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self, inputs, length):
        batch_size = inputs.size()[0]

        embed = self.dropout(self.embedding(inputs))
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed, length, batch_first=True,enforce_sorted=False)
        gru_out, hidden = self.gru(pack_embed)
       
 #         hidden_size [2, batch_size, 512]
        
        
        hidden = torch.cat((hidden[0], hidden[1]), dim = 1)
        hidden = hidden.unsqueeze(0)

        return hidden


# In[ ]:


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors.cuda()
        self.length = len(embedding.vectors)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.gru = nn.GRU(300,1024)
        self.hidden2tag = nn.Linear(1024, self.length)
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self, inputs, hidden):
          
        #inputs size : [1, batch_size]
        inputs = inputs.view(1, inputs.size()[0])
        embed = self.dropout(self.embedding(inputs))
        
        gru_out, hidden = self.gru(embed, hidden)

        output = self.hidden2tag(gru_out)
        output = output.reshape(output.size()[1], output.size()[2])
        
        return output, hidden


# In[ ]:


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.length = len(embedding.vectors)
        
    
    def forward(self, texts, text_length, summary_label):
        batch_size = texts.size()[0]
        
        hidden = self.encoder(texts, text_length)
        
        output_summary = torch.zeros(batch_size, len(summary_label[0]), self.length, dtype = torch.float).cuda()
        
             
        for turn in range(len(summary_label[0])):
            
            #summary_label size : [batch_size, sentence_length]
            inputs = summary_label[:,turn]
            
            #inputs size: [batch_size] (每個batch的第 turn 個字的index， 最開始為<s>的index 1 )
            # hidden size: [1, batch size, 512]
            output, hidden = self.decoder(inputs, hidden)
            
#           #output size : [batch_size, embedding_len]
            
            output_summary[:,turn,:] = output

                
            #output_summary size : [batch_size, sentence_length, embedding_len]
        return output_summary
        


# In[ ]:


encoder = Encoder()
decoder = Decoder()
seq2seq = Seq2seq(encoder, decoder).cuda()


# In[ ]:


import torch.optim as optim
optimizer = optim.Adam(seq2seq.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss(ignore_index = 0)


# In[ ]:


for epoch in range(5):

    count = 0
    for i, data in enumerate(dataloader):
        data_x, data_y = data['text'], data['summary']
        text_length = [(300 if j>300 else j ) for j in data['len_text']]
        
        optimizer.zero_grad()

        data_y = data_y.cuda()
        output = seq2seq(data_x.cuda(), text_length, data_y)
        
        output = output[:,:-1]
        data_y = data_y[:,1:]
        
        output = output.reshape(-1, output.size()[2])
        data_y = data_y.reshape(-1)
        loss = loss_function(output, data_y)
        count = count +1
        if count%400 ==0:
            print(loss)

        loss.backward()
        optimizer.step()

    print(epoch+1,'epoch finish----------')
 


# In[ ]:


torch.save(seq2seq.state_dict(), 'seq2seq_model.pkl') 

