import boto3
import json
'''
Sentiment analysis:

Note, you have to set up the aws command line tool to be able to use this.
I also have it set to run on my personal profile.



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
if __name__ == '__main__':
    session = boto3.session.Session(profile_name='personal')
    comprehend = session.client(
        service_name='comprehend',  region_name='us-west-2')

    text = "I hate you"

    responce = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    print(responce['Sentiment'])
