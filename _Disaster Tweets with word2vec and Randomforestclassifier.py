#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords


# In[65]:


train = pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/train.csv')

test = pd.read_csv('C:/Users/user/Desktop/IVY WORK BOOK/test.csv')


# In[66]:


train.head()


# In[67]:


train.shape


# In[68]:


test.shape


# In[73]:


import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[77]:


def review_wordlist(Inpdata, remove_stopwords=False):
    cleanedArticle1=re.sub(r'[?|$|(),"".@#=><|!]Â&*/',r' ',Inpdata)
    cleanedArticle2=re.sub(r'https?://\S+|www\.\S+',r' ',cleanedArticle1)
    cleanedArticle3=re.sub(r'[^a-z A-Z 0-9]',r' ',cleanedArticle2)
    cleanedArticle4=re.sub(r'\b\w{1,2}\b', ' ',cleanedArticle3)
    review_text=re.sub(r' +', ' ',cleanedArticle4)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if not w in stops]
    return(words)


# In[78]:


def review_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence,remove_stopwords))
        return sentences


# In[79]:


sentences = []
for review in train["text"]:
    sentences += review_sentences(review, tokenizer)


# In[80]:


print(sentences)


# In[81]:


from gensim.models import word2vec
num_features = 300 
min_word_count = 40
num_workers = 4 
context = 10 
downsampling = 1e-3

model = word2vec.Word2Vec(sentences,workers=num_workers,
size=num_features,min_count=min_word_count,window=context,sample=downsampling)

model.init_sims(replace=True)

model_name = "SentimentAnalysis_word2vec"
model.save(model_name)


# In[ ]:





# In[82]:


def featureVecMethod(words, model, num_features):
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[83]:


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs


# In[84]:


clean_train_reviews = []
for review in train['text']:
    clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))
    
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)


# In[85]:


trainDataVecs.shape


# In[86]:


clean_test_reviews = []
for review in test["text"]:
    clean_test_reviews.append(review_wordlist(review,remove_stopwords=True))
    
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


# In[87]:


testDataVecs.shape


# In[88]:


TARGET=['target']
y=train[TARGET].values


# In[92]:


trainDataVecs =np.nan_to_num(trainDataVecs) 


# In[93]:


np.any(np.isnan(trainDataVecs))


# In[94]:


np.all(np.isfinite(trainDataVecs))


# In[95]:


from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[99]:


params={
     
    'n_estimators':[800,700,1200,1500,600,400],
     'max_depth':[4,3,5,6,8,12,15,16,9]
    
}


# In[100]:


metrics.SCORERS.keys()


# In[101]:


from sklearn.ensemble import RandomForestClassifier
#rom sklearn.svm import NuSVC
forest = RandomForestClassifier(criterion='entropy')
forest=GridSearchCV(estimator=forest,param_grid=params,n_jobs=-1,cv=5,scoring='f1')
forest = forest.fit(trainDataVecs,y)
predictions=forest.predict(trainDataVecs)
print('classification report', metrics.classification_report(y,predictions))
print('confusion matrix' ,metrics.confusion_matrix(y,predictions))
print('F1 SCORE' ,metrics.f1_score(y,predictions))


# In[60]:


GV.best_params_


# In[24]:


np.any(np.isnan(testDataVecs))


# In[25]:


testDataVecs =np.nan_to_num(testDataVecs) 


# In[26]:


np.all(np.isfinite(testDataVecs))


# In[62]:


result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id":test["id"], "target":result})
output.to_csv( "output.csv", index=False)


# In[63]:


output


# In[ ]:




