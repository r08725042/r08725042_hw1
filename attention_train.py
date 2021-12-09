
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
    batch_size = 8,
    shuffle = True,
    collate_fn = lambda x: Seq2SeqDataset.collate_fn(train, x)
)


# In[ ]:


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
        self.gru2 = nn.GRU(1024,512, bidirectional=True, batch_first = True)

        self.hidden2hid = nn.Linear(1024, 1024)
        
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self, inputs, length):     
        embed = self.dropout(self.embedding(inputs))
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed, length, batch_first=True,enforce_sorted=False)
        gru_out, hidden = self.gru(pack_embed)
        gru_out, hidden = self.gru2(gru_out)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
 

        #         hidden_size : [2, 32, 256]
        
        
        hidden = torch.tanh(self.hidden2hid(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
#         hidden size : [batch size, max length, 512]
#         output size : [batch size, max length, 1024]
        return output, hidden


# In[ ]:


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        
        self.attention = nn.Linear(1024+1024, 1024)
        self.v = nn.Linear(1024, 1, bias = False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, hidden, e_output, mask):
        batch_size = e_output.size()[0]
        max_length = e_output.size()[1]
        
        hidden = hidden.unsqueeze(1)
        hidden = hidden.repeat(1, max_length, 1)
#         hidden size = [batch size, max_length, 1024]
        
        
        energy = torch.tanh(self.attention(torch.cat((hidden, e_output), dim = 2)))
        
        #energy = [batch size, max_length, 1024]
        
        attention = self.v(energy).squeeze(2)
        #attention size = [batch size, 300]
        
        attention = attention.masked_fill(mask==False, -1e10)
        
        return self.softmax(attention)
        
        


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, attention):
        super(Decoder, self).__init__()

        
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors.cuda()
        
        self.attention = attention
        
        self.length = len(embedding.vectors)
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.gru = nn.GRU(1024+300,1024)
        
        self.hidden2out = nn.Linear(1024+300+1024, self.length)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, inputs, hidden, e_output, mask):
        
        #inputs size : [1, batch_size]
        inputs = inputs.view(1, inputs.size()[0])
        embed = self.dropout(self.embedding(inputs))
        
        a = self.attention(hidden, e_output, mask)
        a = a.unsqueeze(1)
        
#         a size : [batch size, max length]
            
        weight = torch.bmm(a, e_output)
        weight = weight.permute(1, 0, 2)
#         weight size : [1, batch size, 1024]
        
        gru_input = torch.cat((embed, weight), dim = 2)
#         gru_input size : [1, batch size, 1024+300]
        
        hidden = hidden.unsqueeze(0)
        gru_out, hidden = self.gru(gru_input, hidden)
        
        #gru_out size : []?
        embed = embed.squeeze(0)
        gru_out = gru_out.squeeze(0)
        weight = weight.squeeze(0)
        
        prediction = self.hidden2out(torch.cat((embed, gru_out, weight), dim = 1))
#         prediction size : [batch size, embedding length]
        
        hidden = hidden.squeeze(0)
        
        return prediction, hidden


# In[ ]:


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    
    def forward(self, texts, text_length, summary_turn, hidden, mask, e_output):
        
        if e_output.size()[2] == 1:
            e_output, hidden = self.encoder(texts, text_length)

        texts = texts.cpu()
        inputs = summary_turn
        
        output, hidden = self.decoder(inputs, hidden, e_output, mask)
        
        return output, hidden, e_output
                


# In[ ]:


attention = Attention()
encoder = Encoder()
decoder = Decoder(attention)
seq2seq = Seq2seq(encoder, decoder).cuda()

import torch.optim as optim
optimizer = optim.Adam(seq2seq.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss(ignore_index = 0)


# In[ ]:


print('now training...')
for epoch in range(5):

    count = 0
    for i, data in enumerate(dataloader):
        data_x, data_y = data['text'], data['summary']
        data_mask = data['attention_mask']
        text_length = [(300 if j>300 else j ) for j in data['len_text']]
        
        optimizer.zero_grad()

        data_y = data_y.cuda()
        
        # model_input : texts, text_length, summary_turn, hidden, mask, encoder_output
        # initial e_output, hidden, total_loss
        
        e_output = torch.zeros(1,1,1)
        hidden = 0
        total_loss=0
        data_x = data_x.cuda()
        for turn in range(len(data_y[0])-1):
                
            output, hidden, e_output = seq2seq(data_x, text_length, (data_y[:,turn]).cuda(),hidden, data_mask.cuda(), e_output)
            loss = loss_function(output, data_y[:,turn+1])
            total_loss = total_loss + loss
            

        count = count +1
        if count%800 ==0:
            print(total_loss/len(data_y[0]-1))
        
        (total_loss/len(data_y[0]-1)).backward()
        optimizer.step()

    print(epoch+1,'epoch is finish----------')
    
    if (total_loss/len(data_y[0]-1)) < 3.5:
        name = 'attention_model_'+str(epoch+1)+'.pkl'
        torch.save(seq2seq.state_dict(), name) 
    else:
        print('>=3.5')


# In[ ]:


torch.save(seq2seq.state_dict(), 'attention_model.pkl') 

