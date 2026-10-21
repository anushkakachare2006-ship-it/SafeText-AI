#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Lib
import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv("labeled_data.csv")


# In[3]:


dataset


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.describe()


# In[6]:


dataset["labels"]=dataset["class"].map({0:"Hate Speech", 1:"Offensive Language", 2:"Neither offensive nor Hate speech"})


# In[7]:


dataset


# In[8]:


data= dataset[["tweet","labels"]]


# In[9]:


data


# In[10]:


import re
import nltk
import string


# In[11]:


#Importing stop words
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))


# In[12]:


#import steming
stemmer=nltk.SnowballStemmer("english")


# In[13]:


#Data Cleaning
def clean_data(text):
    text=str(text).lower()
    text=re.sub('https?://\S+|www.S+','', text)
    text=re.sub("\[.*?\]",'', text)
    text=re.sub('<.*?>+','', text)
    text=re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=[word for word in text.split(' ') if word not in stopwords]
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


# In[14]:


data["tweet"]=data["tweet"].apply(clean_data)


# In[15]:


data


# In[16]:


x=np.array(data["tweet"])
y=np.array(data["labels"])


# In[17]:


x


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[19]:


cv=CountVectorizer()
x=cv.fit_transform(x)


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[21]:


#Building a ML Model
from sklearn.tree import DecisionTreeClassifier


# In[22]:


dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[23]:


y_pred=dt.predict(x_test)


# In[24]:


# Creating confussion & accuracy matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[26]:


import seaborn as sns
import matplotlib.pyplot as ply


# In[32]:


sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")


# In[28]:


sample = "Let's unite and kill all the people who are protesting against the government"
sample = clean_data(sample)


# In[29]:


data1= cv.transform([sample]).toarray()


# In[30]:


dt.predict(data1)

