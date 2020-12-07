# model.py

import re 
import string
import itertools
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class Model:

    df = None
    tfidf = None
    classifier = None
    score = None
    cf_matrix = None

    def train(self, src):
        self.df=pd.read_csv(src)
        labels=self.df.label
        x_train,x_test,y_train,y_test=train_test_split(self.df['text'], labels, test_size=0.2, random_state=7)
        self.tfidf=TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train=self.tfidf.fit_transform(x_train) 
        tfidf_test=self.tfidf.transform(x_test)
        self.classifier=PassiveAggressiveClassifier(max_iter=50)
        self.classifier.fit(tfidf_train, y_train)
        y_pred=self.classifier.predict(tfidf_test)
        self.score=accuracy_score(y_test, y_pred)
        self.cf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])

    def predict(self, title, text):
        x_test = pd.Series(data=[text], index=[0])
        tfidf_test = self.tfidf.transform(x_test)
        prediction = self.classifier.predict(tfidf_test)
        prediction = prediction[0]
        return prediction

    def preprocess_text(self, text):
        text = text.lower()
        text = text.encode('ascii', 'ignore').decode()
        text = re.sub(r'https*\S+', ' ', text)
        text = re.sub(r'@\S+', ' ', text)
        text = re.sub(r'#\S+', ' ', text)
        text = re.sub(r'\'\w+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\w*\d+\w*', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text