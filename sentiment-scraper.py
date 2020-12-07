import boto3
import json
import pandas as pd
'''
Sentiment analysis:

Note, you have to set up the aws command line tool to be able to use this.
I also have it set to run on my personal profile.

Also we only get 50k of text, and 5M characters.

This is what the json looks like:

{
    "ResponseMetadata": {
        "HTTPHeaders": {
            "content-length": "164",
            "content-type": "application/x-amz-json-1.1",
            "date": "Tue, 24 Nov 2020 22:19:39 GMT",
            "x-amzn-requestid": "b9229ee9-a7f5-48a7-9977-68e3e4701fa5"
        },
        "HTTPStatusCode": 200,
        "RequestId": "b9229ee9-a7f5-48a7-9977-68e3e4701fa5",
        "RetryAttempts": 0
    },
    "Sentiment": "NEGATIVE",
    "SentimentScore": {
        "Mixed": 0.007010723929852247,
        "Negative": 0.978324830532074,
        "Neutral": 0.01324869878590107,
        "Positive": 0.0014157623518258333
    }
}


'''


def test_cvs():
    session = boto3.session.Session(profile_name='personal')
    comprehend = session.client(
        service_name='comprehend',  region_name='us-west-2')

    test = pd.read_csv('train.csv')

    # train = pd.read_csv('train.csv')

    sentiments = []

    for idx, row in test.iterrows():
        text = row['text'][:4000]
        responce = comprehend.detect_sentiment(Text=text, LanguageCode='en')
        sentiment = responce['Sentiment']
        sentiments.append(sentiment)

    df = pd.DataFrame(sentiments)
    df.to_csv("train_sentiments.csv")

    """ for idx, row in train.iterrows():
        print(row['text'][:4000])
        break """


def news_scraper():
    session = boto3.session.Session(profile_name='personal')
    comprehend = session.client(
        service_name='comprehend',  region_name='us-west-2')
    """
    text = "I hate you"

    responce = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    print(responce['Sentiment']) """

    sentiments = ['Negative', 'Nuetral']

    df = pd.read_csv('news.csv')
    for idx, row in df.iterrows():
        text = row['text'][:4000]
        responce = comprehend.detect_sentiment(Text=text, LanguageCode='en')
        sentiment = responce['Sentiment']
        sentiments.append(sentiment)

    df = pd.DataFrame(sentiments)
    df.to_csv("news_sentiments.csv")


def merge_train():
    print("Merging datasets.. ")
    train = pd.read_csv("train.csv")
    sentiments = pd.read_csv("train_sentiments.csv")
    print(sentiments['0'])

    train['Sentiment'] = sentiments['0']

    train.to_csv("train_with_sentiment.csv")
    return


def merge_test():
    print("Merging datasets.. ")
    test = pd.read_csv("test.csv")
    sentiments = pd.read_csv("test_sentiments.csv")
    print(test.shape)
    print(sentiments.shape)
    test['Sentiment'] = sentiments['0']
    print(test)

    test.to_csv("test_with_sentiment.csv")
    return


def generate_sentiment_scrapper():
    session = boto3.session.Session(profile_name='personal')
    comprehend = session.client(
        service_name='comprehend',  region_name='us-west-2')
    """
    text = "I hate you"

    responce = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    print(responce['Sentiment']) """

    sentiments = []

    df = pd.read_csv('alldata.csv')
    print(df.shape)
    for idx, row in df.iterrows():
        text = row['text'][:4000]
        responce = comprehend.detect_sentiment(Text=text, LanguageCode='en')
        #sentiment = responce['Sentiment']
        # sentiments.append(sentiment)
        pos = responce['SentimentScore']['Positive']
        neg = responce['SentimentScore']['Negative']
        neu = responce['SentimentScore']['Neutral']
        mix = responce['SentimentScore']['Mixed']
        sentiments.append([pos, neg, neu, mix])

    df = pd.DataFrame(sentiments)
    df.to_csv("testing_sentiments.csv")


def merge_all_data_sentiment():
    all_data = pd.read_csv('alldata.csv')
    sentiment = pd.read_csv("alldata_sentiment.csv")
    pos_col = sentiment['0']
    neg_col = sentiment['1']
    nue_col = sentiment['2']
    mix_col = sentiment['3']

    print(pos_col)
    print(mix_col)
    all_data['positive'] = pos_col
    all_data['negative'] = neg_col
    all_data['neutral'] = nue_col
    all_data['mixed'] = mix_col

    all_data.to_csv("alldata_with_sentiment.csv")


if __name__ == '__main__':
    merge_all_data_sentiment()
