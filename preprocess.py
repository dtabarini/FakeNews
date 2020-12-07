import pandas as pd 
import re 
import string

def text_preproc(x):
  x = x.lower()
  x = x.encode('ascii', 'ignore').decode()
  x = re.sub(r'https*\S+', ' ', x)
  x = re.sub(r'@\S+', ' ', x)
  x = re.sub(r'#\S+', ' ', x)
  x = re.sub(r'\'\w+', '', x)
  x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
  x = re.sub(r'\w*\d+\w*', '', x)
  x = re.sub(r'\s{2,}', ' ', x)
  return x

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data =[train, test]
feats = ['statement','text']
for df in data:
    for feat in feats:
        df[feat] = df[feat].apply(text_preproc)



test.to_csv("test_processed.csv", index = False)
train.to_csv("train_processed.csv", index = False)


