#!/usr/bin/env python
# coding: utf-8

# In[1]:



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import pickle


# In[2]:


with open('a_w.pkl', 'rb') as f:
    a_w = pickle.load(f)

# a_w file include text and attention weight
# In[3]:


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


# In[ ]:





# In[53]:


attention = []
for i in (a_w[0]['a'][0][:19]):
    attention.append(i[:19])


# In[54]:


temp_input = a_w[0]['text'][0].split(' ')


# In[55]:


temp_input = temp_input[:19]
output = []
for i in (a_w[0]['predict'][0].split(' ')):
    output.append(i)


# In[56]:


inputs = ''
for i in temp_input:
    inputs = inputs+i+' '
inputs = inputs[:-1]


# In[57]:


showAttention(inputs, output,attention)

