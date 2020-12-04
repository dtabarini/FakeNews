# model.py

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class Model:
    df = None
    labels = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    tfidf_vectorizer = None
    tfidf_train = None
    tfidf_test = None
    pac = None
    y_pred = None
    score = None
    cf_matrix = None

    def initialize(self, src):
        self.df=pd.read_csv(src)
        self.labels=self.df.label
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.df['text'], self.labels, test_size=0.2, random_state=7)
        self.tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
        self.tfidf_train=self.tfidf_vectorizer.fit_transform(self.x_train) 
        self.tfidf_test=self.tfidf_vectorizer.transform(self.x_test)
        self.pac=PassiveAggressiveClassifier(max_iter=50)
        self.pac.fit(self.tfidf_train,self.y_train)
        self.y_pred=self.pac.predict(self.tfidf_test)
        self.score=accuracy_score(self.y_test,self.y_pred)
        self.cf_matrix = confusion_matrix(self.y_test,self.y_pred, labels=['FAKE','REAL'])

    def predict(self, title, text):
        new_x_test = pd.Series(data=[text], index=[0])
        new_tfidf_test = self.tfidf_vectorizer.transform(new_x_test)
        prediction = self.pac.predict(new_tfidf_test)
        return prediction[0]
