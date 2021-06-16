#!/usr/bin/env python
# coding: utf-8

# In[5]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation


# In[18]:


stopwords=list(STOP_WORDS)


# In[19]:


nlp =spacy.load('en_core_web_sm')


# In[20]:


text ="Papaya is a healthy fruit with a list of properties that is long and exhaustive. You can munch on it as a salad, have it cooked or boiled or just drink it up as milkshake or juices. Papaya has many virtues that can contribute to our good health. The most important of these virtues is the protein-digesting enzyme it has. The enzyme is similar to pepsin in its digestive action and is said to be so powerful that it can digest 200 times its own weight in protein. It assists the body in assimilating the maximum nutritional value from food to provide energy and bodybuilding materials.Papain in raw papaya makes up for the deficiency of gastric juice and fights excess of unhealthy mucus in the stomach and intestinal irritation. The ripe fruit, if eaten regularly corrects habitual constipation, bleeding piles and chronic diarrhea. The juice of the papaya seeds also assists in the above-mentioned ailments. Papaya juice, used as a cosmetic, removes freckles or brown spots due to exposure to sunlight arid makes the skin smooth and delicate. A paste of papaya seeds is applied in skin diseases like those caused by ringworm.The black seeds of the papaya are highly beneficial in the treatment of cirrhosis of the liver caused by alcoholism, malnutrition, etc. A tablespoonful of its juice, combined with a hint of fresh lime juice, should be consumed once or twice daily for a month. The fresh juice of raw papaya mixed with honey can be applied over inflamed tonsils, for diphtheria and other throat disorders.It dissolves the membrane and prevents infection from spreading "


# In[21]:


doc=nlp(text)


# In[22]:


tokens=[token.text for token in doc]
print(tokens)


# In[23]:


punctuation=punctuation +'\n'
punctuation


# In[25]:


word_frequencies={}
for word in doc:
     if word.text.lower() not in stopwords:
        if  word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text]=1
            else:
                word_frequencies[word.text]+=1
                


# In[27]:


print(word_frequencies)


# In[28]:


max_frequency = max(word_frequencies.values())


# In[29]:


max_frequency


# In[30]:


for word in word_frequencies.keys():
    word_frequencies[word]= word_frequencies[word]/max_frequency
    


# In[31]:


print(word_frequencies)


# In[32]:


sentence_tokens=[sent for sent in doc.sents]


# In[33]:


sentence_tokens


# In[34]:


sentence_scores={}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent]=word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] +=word_frequencies[word.text.lower()]
                


# In[35]:


sentence_scores


# In[36]:


from heapq import nlargest


# In[38]:


select_length=int(len(sentence_tokens)*0.20)


# In[39]:


select_length


# In[40]:


summary=nlargest(select_length,sentence_scores,key=sentence_scores.get)


# In[41]:


summary


# In[42]:


final_summary=[word.text for word in summary]


# In[44]:


summary=' '.join(final_summary)


# In[45]:


summary


# In[46]:


len(text)


# In[47]:


len(summary)


# In[ ]:




