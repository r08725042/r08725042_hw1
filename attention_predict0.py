
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

from torch.utils.data import Dataset
from dataset import *
from utils import *
from argparse import ArgumentParser


# In[ ]:


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()


# In[ ]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


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


from preprocess_seq2seq import *
print('data preprocessing...')
logging.info('Creating test dataset...')
create_seq2seq_dataset(
    process_samples(tokenizer, test),
    'abstractive_test.pkl',
    tokenizer.pad_token_id
)


# In[ ]:


file = open('abstractive_test.pkl','rb')    
test = pickle.load(file)      
file.close()


# In[ ]:


validloader = DataLoader(
    dataset = test,
    batch_size = 16,
    shuffle = False,
    collate_fn = lambda x: Seq2SeqDataset.collate_fn(test, x)
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

        self.hidden2hid = nn.Linear(1024, 1024)
        
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self, inputs, length):     
        embed = self.dropout(self.embedding(inputs))
        
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(embed, length, batch_first=True,enforce_sorted=False)
        gru_out, hidden = self.gru(pack_embed)
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
        with open('embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.length = len(embedding.vectors)
        
    def forward(self, texts, text_length, mask):
        batch_size = texts.size()[0]
        
        
        e_output, hidden = self.encoder(texts, text_length)
        output_summary = torch.zeros(batch_size, 80, 1, dtype = torch.float).cuda()
        inputs = torch.ones(batch_size, dtype = torch.long).cuda()
             
        for turn in range(40):
            
            
            
            #inputs size: [batch_size] (每個batch的第 turn 個字的index， 最開始為<s>的index 1 )
            # hidden size: [1, batch size, 512]
            output, hidden = self.decoder(inputs, hidden, e_output, mask)
            
#           #output size : [batch_size, embedding_len]
            output = output.argmax(1)
            inputs = output
            output = output
        
            output = output.reshape(output.size()[0], 1)
            output_summary[:,turn,:] = output
            
            
            

                
            #output_summary size : [batch_size, sentence_length, 1]
        return output_summary
                
        


# In[ ]:


print('load model weight...')
attention = Attention()
encoder = Encoder()
decoder = Decoder(attention)
seq2seq = Seq2seq(encoder, decoder).cuda()


# In[ ]:


seq2seq.load_state_dict(torch.load('attention_model.pkl'))


# In[ ]:


print('now predicting...')
predict_id = []
predict_sentence = []

for i, valid_data in enumerate(validloader):
    for ids in valid_data['id']:
        predict_id.append(ids)
    
    valid_data_x = valid_data['text']
    text_length = [(300 if j>300 else j ) for j in valid_data['len_text']]
    valid_data_mask = valid_data['attention_mask']
    seq2seq.eval()

    
    output = seq2seq(valid_data_x.cuda(), text_length, valid_data_mask.cuda())
    
    output = output.reshape(output.size()[0], output.size()[1])
      
        

    for s in output:
        predict_sentence.append(s)


# In[ ]:


print('create predict file')


# In[ ]:


token = Tokenizer()
with open('embedding.pkl', 'rb') as f:
    
    embedding = pickle.load(f)
    embedding_weight = embedding.vectors
token.set_vocab(embedding.vocab)


# In[ ]:


predict = []

for p_sentence in predict_sentence:
    temp = []
   
    for i in p_sentence:
        if int(i) == 2:
            break
        elif int(i) == 3:
            continue
        else:
            temp.append(int(i))
            
    predict.append(token.decode(temp))


# In[ ]:


data = []
for i in range(len(predict)):

    temp = {'id':predict_id[i], 'predict' : predict[i]}
    data.append(temp)
    
output_file = Path(args.output_path)
output_file.write_text(
    '\n'.join(
        [
            json.dumps(line) for line in data
        ]
    ) + '\n'
)

