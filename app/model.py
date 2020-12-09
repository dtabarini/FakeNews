# model.py

import re 
import string
import itertools
import statistics
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import bert.tokenization as tokenization
from numpy import var, std
from statistics import mean
from sklearn.svm import LinearSVC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, model_from_json
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

    bert_df = None
    bert_layer = None
    bert_tokenizer = None
    bert_model = None
    bert_status = 0

    def bert_init(self):
        if self.bert_status != 0:
            raise ValueError("BERT not uninitialized")
            return False
        bert_module = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
        self.bert_layer = hub.KerasLayer(bert_module, trainable=True)
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self.bert_status = 1

    def bert_load(self, src):
        if self.bert_status != 1:
            raise ValueError("BERT not initialized")
            return False
        json_file = open('bert/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'KerasLayer': hub.KerasLayer})
        loaded_model.load_weights("bert/model.h5")
        self.bert_model = loaded_model
        self.bert_status = 2

    def bert_train(self, src, output=True):
        if self.bert_status != 1:
            raise ValueError("BERT not initialized")
            return False
        self.bert_df = pd.read_csv(src)
        df_test = self.bert_df[-2000:]
        df_train = self.bert_df[:-2000]
        df_fake = df_train[df_train['label'] == 'fake']
        df_real = df_train[df_train['label'] == 'real']
        df_real = df_real.sample(n=7740, random_state=SEED)
        df_balanced = df_real.append(df_fake)
        df_balanced = df_balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df_balanced.label = df_balanced.label.replace({'real': 0, 'fake': 1})
        train_input = self.bert_encode(df_balanced.statement.values, self.bert_tokenizer)
        train_labels = df_balanced.label.values
        self.bert_model = self.build_model(bert_layer)
        train_history = self.bert_model.fit(
            train_input, train_labels,
            validation_split=0.2,
            epochs=2, batch_size=5
        )
        if output:
            # output model json
            model_json = self.bert_model.to_json()
            with open("bert/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.bert_model.save_weights("bert/model.h5")
        self.bert_status = 2

    def predict(self, title, text, model_type):
        model = None
        test_data = None
        prediction = None
        if model_type == "bert":
            if self.bert_status != 2:
                raise ValueError("BERT not loaded or trained")
                return None
            model = self.bert_model
            test_data = self.bert_encode([title], self.bert_tokenizer)
            prediction = model.predict(test_data)
            prediction = "fake" if (prediction[0][0] > 0.5) else "real"
        else:
            if model_type == "logreg":
                model = self.classifier_logreg
            elif model_type == "linsvc":
                model = self.classifier_linsvc
            else:
                raise ValueError("invalid argument: model_type")
                return None
            x_test = pd.Series(data=[self.preprocess_text(text)], index=[0])
            test_data = self.tfidf.transform(x_test)
            prediction = model.predict(test_data)
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

    # BERT helper function for encoding input texts
    def bert_encode(self, texts, tokenizer, max_len=512):
        all_tokens = []
        all_masks = []
        all_segments = []
        
        for text in texts:
            text = tokenizer.tokenize(text)
                
            text = text[:max_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_len - len(input_sequence)
            
            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_len
            
            all_tokens.append(tokens)
            all_masks.append(pad_masks)
            all_segments.append(segment_ids)
        
        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    # BERT helper function for making the NN
    def build_model(self, bert_layer, max_len=512):
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)
        
        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model





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