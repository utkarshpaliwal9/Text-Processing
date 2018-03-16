# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:28:18 2018

@author: Utkarsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import os
#os.chdir('C:\Users\alien\Documents\GitHub\Text-Processing')

dataset = pd.read_csv('Restaurant_Reviews.tsv', sep = '\t', quoting = 3)

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #sub function to remove letters other than alphabets
    review = review.lower();
    review = review.split();
    ps = PorterStemmer()
    #stemming the words not in stopwords and adding to the review
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating Bag of Words    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Fitting Naive Bayes to the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)