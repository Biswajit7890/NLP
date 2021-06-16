#!/usr/bin/env python
# coding: utf-8

# In[247]:


import spacy
import pickle 
import random


# In[248]:


train=pickle.load(open('train_data.pkl', 'rb'))


# In[249]:


train


# In[250]:


nlp = spacy.blank("en")

def train_model(train):
    if'ner' not in nlp.pipe_names:
        ner=nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        
    for _,annotation in train:
        for ent in annotation['entities']:
             ner.add_label(ent[2])
            
    other_pipes=[pipe for pipe in nlp.pipe_names if pipe!='ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer=nlp.begin_training()
        for it in range(100):
            print("Iteration :"+str(it))
            random.shuffle(train)
            losess={}
            index =0
            for text, annotations in train:
                try:
                    nlp.update([text],[annotations],drop=0.2,sgd=optimizer,losess=losess)
                except Exception as e:
                    pass
            print(losess)
           
            


# In[251]:


train_model(train)


# In[252]:


train[10][0]


# In[253]:


nlp.to_disk('nlp_model')


# In[254]:


nlp_model=spacy.load('nlp_model')


# In[ ]:





# In[255]:


doc=nlp_model(train[5][0])
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}-{ent.text}')


# In[256]:


import sys, fitz
fname="C:/Users/user/Downloads/resume/Alice Clark CV.pdf"
doc=fitz.open(fname)
text=""
for page in doc:
    text=text+str(page.getText())
tx=" ".join(text.split('\n'))
print(tx)


# In[257]:


##!pip install pyMuPDF


# In[258]:


doc=nlp_model(tx)
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}-{ent.text}')


# In[ ]:




