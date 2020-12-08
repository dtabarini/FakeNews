# model.py

import re 
import string
import itertools
import statistics
import numpy as np
import pandas as pd
from numpy import var, std
from statistics import mean
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression

SEED = 42

class ModelV2:

    df = None
    tfidf = None
    classifier_logreg = None
    classifier_linsvc = None

    def train(self, src):
        self.df=pd.read_csv(src)
        labels=self.df.label
        x_df = self.df.drop('label', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x_df, labels, train_size=0.8, test_size=0.2, random_state=SEED)
        self.tfidf, tfidf_train, tfidf_test = self.convert_to_tfidf_train_and_test_sets_ret(x_train, x_test)
        self.classifier_logreg = LogisticRegression(penalty='elasticnet', l1_ratio=1, C=1, solver='saga', max_iter=1e4, n_jobs=-1)
        self.classifier_logreg.fit(tfidf_train, y_train)
        # aucScore, stdDev, scoresArray = self.get_cv_score(self.classifier_logreg, x_df, labels, iterations=5, get_details=True)
        # conf_matrix = self.get_confusion_matrix(self.classifier_logreg, x_df, labels)
        self.classifier_linsvc = LinearSVC(class_weight='balanced', penalty='l1', dual=False, C=0.3, max_iter=1e4, verbose=1)
        self.classifier_linsvc.fit(tfidf_train, y_train)
        # aucScore, stdDev, scoresArray = self.get_cv_score(self.classifier_linsvc, x_df, labels, iterations=5, get_details=True)
        # conf_matrix = self.get_confusion_matrix(self.classifier_linsvc, x_df, labels)

    def predict(self, title, text, model):
        x_test = pd.Series(data=[text], index=[0])
        tfidf_test = self.tfidf.transform(x_test)
        classifier = None
        if model == "logreg":
            classifier = self.classifier_logreg
        elif model == "linsvc":
            classifier = self.classifier_linsvc
        else:
            raise ValueError("missing argument: model")
        prediction = classifier.predict(tfidf_test)
        prediction = prediction[0]
        return prediction


    # Define helper functions here

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

    def convert_series_to_integer_labels(self, y_train):
        labels = y_train.to_numpy(dtype='str')
        for i in range(len(labels)):
            if labels[i].lower() == 'fake':
                labels[i] = 1
            elif labels[i].lower() == 'real':
                labels[i] = 0
            else:
                print(y_train)
                raise NameError("y_train data contains labels that are neither 'fake' nor 'real'!")
        return labels.astype(np.int8)

    def convert_predictions_to_integer_labels(self, predictions):
        # Passed in predictions should be numpy arrays of np.str_ type
        for i in range(len(predictions)):
            if predictions[i].lower() == 'fake':
                predictions[i] = 1
            elif predictions[i].lower() == 'real':
                predictions[i] = 0
            else:
                raise NameError("y_train data contains labels that are neither 'fake' nor 'real'!")
        return predictions.astype(np.int8)

    def convert_to_tfidf_train_and_test_sets(self, x_train, x_test):
        """ Returns a tuple containing the converted tfidf vectors: (tfidf_train, tfidf_test)"""
        # Initialize a TfidfVectorizer to filter out English stop words of the most common words and vectorize article text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train['text'])
        tfidf_test = tfidf_vectorizer.transform(x_test['text'])
        return (tfidf_train, tfidf_test)

    def convert_to_tfidf_train_and_test_sets_ret(self, x_train, x_test):
        """ Returns a tuple containing the converted tfidf vectors: (tfidf_train, tfidf_test)"""
        # Initialize a TfidfVectorizer to filter out English stop words of the most common words and vectorize article text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(x_train['text'])
        tfidf_test = tfidf_vectorizer.transform(x_test['text'])
        return (tfidf_vectorizer, tfidf_train, tfidf_test)

    def get_cv_score(self, model, x_df, labels, iterations=5, get_details=False, train_split_size=0.80):
        """ 
            Pass in an untrained model, x_df, and labels to return the average accuracy score across iterations (5 by default).
            Note if get_details is set to True, this method returns a tuple in the format:
            (mean, std, scores_array)
        """
        auc_scores=[]
        
        for i in range(iterations):
            x_train1, x_test1, y_train1, y_test1 = train_test_split(x_df, labels, train_size=train_split_size, test_size=1-train_split_size, random_state=i)
            tfidf_train, tfidf_test = self.convert_to_tfidf_train_and_test_sets(x_train1, x_test1)
            model.fit(tfidf_train, y_train1)
            true_binary_labels = self.convert_series_to_integer_labels(y_test1)
            predicted_binary_labels = self.convert_predictions_to_integer_labels(model.predict(tfidf_test))
            auc_scores.append(roc_auc_score(true_binary_labels, predicted_binary_labels))
            
        if get_details:
            return (statistics.mean(auc_scores), np.std(auc_scores), auc_scores)
        else:
            return statistics.mean(auc_scores)

    def get_confusion_matrix(self, model, x_df, labels):
        """ Given an untrained model and the true labels"""
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x_df, labels, train_size=0.8, test_size=0.2, random_state=SEED)
        tfidf_train, tfidf_test = self.convert_to_tfidf_train_and_test_sets(x_train1, x_test1)
        model.fit(tfidf_train, y_train1)
        # Visualize the confusion matrix to gain insight into false postives and negatives
        predictions = model.predict(tfidf_test)
        return confusion_matrix(y_test1, predictions, labels=['fake','real'])





class ModelV1:

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