#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk 
nltk.download('stopwords')
import nltk 
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA


# In[44]:


def loadData(trainingFile, testingFile):
    
    with open(trainingFile, "r") as fr1:
        trainFile = fr1.readlines()
    
    with open(testingFile, "r") as fr2:
        testFile = fr2.readlines()
    
 
    train_sentiments_t = [x.split("\t", 1)[0] for x in trainFile]
    train_reviews_t = [x.split("\t", 1)[1] for x in trainFile]
    
    return train_reviews_t, testFile, train_sentiments_t


# In[45]:


train_reviews, test_reviews, train_sentiments = loadData('data/train.dat', 'data/test.dat')


# In[46]:


def clean(reviews):
    
    
    clean_train_reviews = []

    
    for index, review in enumerate(reviews):
        clean_train_reviews.append(preProcess(review))
    
    return clean_train_reviews


# In[47]:


def preProcess(rawReview):

    text_only = BeautifulSoup(rawReview).get_text()
    
    noEmail = re.sub(r'([\w\.-]+@[\w\.-]+\.\w+)','',text_only)
    
    noUrl = re.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|         [a-z0-9.\-]+[.][a-z]{2,4}/|[a-z0-9.\-]+[.][a-z])(?:[^\s()<>]+|\(([^\s()<>]+|         (\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))','', noEmail)
    
    
    smileys = """:-) :) :o) :D :-D :( :-( :o(""".split()
    smileyPattern = "|".join(map(re.escape, smileys))
    
    letters_only = re.sub("[^a-zA-Z" + smileyPattern + "]", " ", noUrl)
    
    words = letters_only.lower().split()     
    
    stops = set(stopwords.words("english"))                  
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = ''
    for word in words:
        if word not in stops and len(word) > 3:
        
            lemmatized_words += str(lemmatizer.lemmatize(word)) + ' '
    
    return lemmatized_words


# In[48]:


def createTFIDFMatrices(train_data, test_data):
    
    vectorizer = TfidfVectorizer(norm = 'l2')
    
    train_matrix = vectorizer.fit_transform(train_data)
    
    test_matrix = vectorizer.transform(test_data)

    return train_matrix, test_matrix


# In[49]:


def csr_l2normalize(mat, copy=False, **kargs):
    
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# In[50]:


def findSimilarities(train_matrix, test_matrix):
    
    cosineSimilarities = np.dot(test_matrix, np.transpose(train_matrix))
    similarities = cosineSimilarities.toarray()
        
    return similarities


# In[51]:


def findKNearest(similarity_vector, k):
    return np.argsort(-similarity_vector)[:k]


# In[52]:


def predict(nearestNeighbors, labels):
    positiveReviewsCount = 0
    negativeReviewsCount = 0
    for neighbor in nearestNeighbors:
        if int(labels[neighbor]) == 1:
            positiveReviewsCount += 1
        else:
            negativeReviewsCount += 1
    if positiveReviewsCount > negativeReviewsCount:
        return 1
    else:
        return -1


# In[53]:


train_reviews = clean(train_reviews)
test_reviews = clean(test_reviews)

train_matrix, test_matrix = createTFIDFMatrices(train_reviews, test_reviews)
train_matrix_norm = csr_l2normalize(train_matrix, copy=True)
test_matrix_norm = csr_l2normalize(test_matrix, copy=True)


# In[54]:


similarities = findSimilarities(train_matrix_norm, test_matrix_norm)


# In[55]:


k = 500
test_sentiments = list()

for similarity in similarities:
    knn = findKNearest(similarity, k)
    prediction = predict(knn, train_sentiments)
    
    
    if prediction == 1:
        test_sentiments.append('+1')
    else:
        test_sentiments.append('-1')


# In[56]:


output = open('output.dat', 'w')

output.writelines( "%s\n" % item for item in test_sentiments )

output.close()


# In[ ]:




