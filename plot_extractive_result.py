
# coding: utf-8

# In[1]:


import json
import pickle


# In[2]:


file = open('extractive_valid.pkl','rb')    
valid = pickle.load(file)      
file.close()

with open('./output/r08725042.ext.jsonl') as f:
    predict = [json.loads(line) for line in f]


# In[3]:


sentence_num=[]
for i in valid:
    sentence_num.append(len(i['sent_range']))


# In[4]:


distribute=[]
for i in range(20000):
    for p in predict[i]['predict_sentence_index']:
        distribute.append(format(p/sentence_num[i],'.1f'))


# In[5]:


count = [0,0,0,0,0,0,0,0,0,0,0]

for i in distribute:
    if i == '1.0':
        count[-1] = count[-1]+1
    else:
        temp = int(i[2])
        count[temp] = count[temp] + 1


# In[6]:


count = [i/40000 for i in count]


# In[7]:



import matplotlib.pyplot as plt
x = [0,1,2,3,4,5,6,7,8,9,10]
y = count
plt.xticks(x, ('0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'))
plt.bar(x, y)


# In[8]:


plt.show()

