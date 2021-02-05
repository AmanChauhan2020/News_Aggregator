from django.shortcuts import render

# Create your views here.

import requests
import nltk
from bs4 import BeautifulSoup
from newspaper import Article
from textblob import TextBlob
import geocoder
import json
import pandas as pd
import numpy as np
import time
#to scrape Twitter
import tweepy
from tweepy import OAuthHandler
import csv
import matplotlib.pyplot as plt
import io
import urllib,base64
import operator
nltk.download('punkt')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import os

#warning    
import warnings
warnings.filterwarnings('ignore')

g = geocoder.ip('me')
lat=str(g.latlng[0])
lon=str(g.latlng[1])
city=str(g[0])
city=city[1:-3]+"India"

api_key = "96d2e493c472b35ec2a8976bdb6d83bd"
base_url = "http://api.openweathermap.org/data/2.5/weather?"

complete_url = base_url + "appid=" + api_key + "&lat=" + lat + "&lon="  + lon
response = requests.get(complete_url) 
x = response.json() 
if x["cod"] != "404": 
  
    y = x["main"] 
  
    current_temperature = y["temp"] 
  
    current_pressure = y["pressure"] 
  
    current_humidiy = y["humidity"] 
  
    z = x["weather"] 
  
    weather_description = z[0]["description"] 
    
    w_temp=str(current_temperature)
    w_press=str(current_pressure)
    w_hum=str(current_humidiy)
    w_desc=str(weather_description.title())
    

def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

        
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

all_pos_words = get_all_words(positive_cleaned_tokens_list)

freq_dist_pos = FreqDist(all_pos_words)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)
   
def sentiment(txt):
    tokens = remove_noise(word_tokenize(txt))
    p=str(classifier.classify(dict([token, True] for token in tokens)))
    if p=="Negative":
        p=-1
        return p
    elif p=="Positive":
        p=1
        return p


pos=[]
pos_link=[]
pos_img=[]
pos_art=[]

neg=[]
neg_link=[]
neg_img=[]
neg_art=[]

neu=[]
neu_link=[]
neu_img=[]
neu_art=[]

headline=[]

web="https://www.indiatoday.in"
web1="https://timesofindia.indiatimes.com/briefs"


it_r = requests.get("https://www.indiatoday.in/world")
it_soup = BeautifulSoup(it_r.content, 'html5lib')
it_headings = it_soup.find_all('h2')
it_headings = it_headings[0:4]
it=[]
for hm in it_headings:
    it.append(hm.text)
    headline.append(hm.text)

    
link=[]
for l in range(0,len(it_headings)): 
    txt=str(it_headings[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link.append(web+new)    
art=[]
img=[]
for i in range(0,len(link)):
    article=Article(link[i])
    article.download()
    article.parse()
    article.nlp()
    img.append(article.top_image)
    art.append(article.summary)
    
    
for i in range(0,len(it)):
    edu=TextBlob(it[i])
    x=edu.sentiment.polarity
    z=sentiment(it[i])
    if x==0:
        neu.append(it[i])
        neu_link.append(link[i])
        neu_img.append(img[i])
        neu_art.append(art[i])
        continue
    if z==1:
        pos.append(it[i])
        pos_link.append(link[i])
        pos_img.append(img[i])
        pos_art.append(art[i])
    elif z==-1:
        neg.append(it[i])
        neg_link.append(link[i])
        neg_img.append(img[i])
        neg_art.append(art[i])
        
        
    
it=tuple(zip(it,link,img,art))




toi_r = requests.get("https://timesofindia.indiatimes.com/briefs/world")
toi_soup = BeautifulSoup(toi_r.content, 'html5lib')
toi_headings = toi_soup.find_all('h2')
toi_headings = toi_headings[2:6]
toi=[]
for mhm in toi_headings:
    toi.append(mhm.text)
    headline.append(mhm.text)
link1=[]
for l in range(0,len(toi_headings)): 
    txt=str(toi_headings[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link1.append(web1+new)

art1=[]
img1=[]
for i in range(0,len(link1)):
    article=Article(link1[i])
    article.download()
    article.parse()
    article.nlp()
    img1.append(article.top_image)
    art1.append(article.summary)
for i in range(0,len(toi)):
    edu=TextBlob(toi[i])
    x=edu.sentiment.polarity
    z=sentiment(toi[i])
    if x==0:
        neu.append(toi[i])
        neu_link.append(link1[i])
        neu_img.append(img1[i])
        neu_art.append(art1[i])
        continue
    if z==1:
        pos.append(toi[i])
        pos_link.append(link1[i])
        pos_img.append(img1[i])
        pos_art.append(art1[i])
        
    elif z==-1:
        neg.append(toi[i])
        neg_link.append(link1[i])
        neg_img.append(img1[i])
        neg_art.append(art1[i])
        
    
        
toi=tuple(zip(toi,link1,img1,art1))



ie_r = requests.get("https://indianexpress.com/section/world/")
ie_soup = BeautifulSoup(ie_r.content, 'html5lib')
ie_headings = ie_soup.find_all('h3')
ie_headings = ie_headings[0:4]
ie=[]
for ph in ie_headings:
    ie.append(ph.text.strip())
    headline.append(ph.text.strip())
link2=[]
for l in range(0,len(ie_headings)): 
    txt=str(ie_headings[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link2.append(new)    
art2=[]
img2=[]
for i in range(0,len(link2)):
    article=Article(link2[i])
    article.download()
    article.parse()
    article.nlp()
    img2.append(article.top_image)
    art2.append(article.summary) 
for i in range(0,len(ie)):
    edu=TextBlob(ie[i])
    x=edu.sentiment.polarity
    z=sentiment(ie[i])
    if x==0:
        neu.append(ie[i])
        neu_link.append(link2[i])
        neu_img.append(img2[i])
        neu_art.append(art2[i])
        continue
    if z==1:
        pos.append(ie[i])
        pos_link.append(link2[i])
        pos_img.append(img2[i])
        pos_art.append(art2[i])
        
    elif z==-1:
        neg.append(ie[i])
        neg_link.append(link2[i])
        neg_img.append(img2[i])
        neg_art.append(art2[i])
        
    
        
ie=tuple(zip(ie,link2,img2,art2))     
    
    
it_r1 = requests.get("https://www.indiatoday.in/india")
it_soup1 = BeautifulSoup(it_r1.content, 'html5lib')
it_headings1 = it_soup1.find_all('h2')
it_headings1 = it_headings1[0:4]
it1=[]
for hm in it_headings1:
    it1.append(hm.text)
    headline.append(hm.text)
link3=[]
for l in range(0,len(it_headings1)): 
    txt=str(it_headings1[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link3.append(web+new)    

art3=[]
img3=[]
for i in range(0,len(link3)):
    article=Article(link3[i])
    article.download()
    article.parse()
    article.nlp()
    img3.append(article.top_image)
    art3.append(article.summary)
for i in range(0,len(it1)):
    edu=TextBlob(it1[i])
    x=edu.sentiment.polarity
    z=sentiment(it1[i])
    if x==0:
        neu.append(it1[i])
        neu_link.append(link3[i])
        neu_img.append(img3[i])
        neu_art.append(art3[i])
        continue
    if z==1:
        pos.append(it1[i])
        pos_link.append(link3[i])
        pos_img.append(img3[i])
        pos_art.append(art3[i])
        
    elif z==-1:
        neg.append(it1[i])
        neg_link.append(link3[i])
        neg_img.append(img3[i])
        neg_art.append(art3[i])
        
    
it1=tuple(zip(it1,link3,img3,art3))    
    


toi_r1 = requests.get("https://timesofindia.indiatimes.com/briefs/india")
toi_soup1 = BeautifulSoup(toi_r1.content, 'html5lib')
toi_headings1 = toi_soup1.find_all('h2')
toi_headings1 = toi_headings1[2:6]
toi1=[]
for mhm in toi_headings1:
    toi1.append(mhm.text)
    headline.append(mhm.text)
link4=[]
for l in range(0,len(toi_headings1)): 
    txt=str(toi_headings1[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link4.append(web1+new)

art4=[]
img4=[]
for i in range(0,len(link4)):
    article=Article(link4[i])
    article.download()
    article.parse()
    article.nlp()
    img4.append(article.top_image)
    art4.append(article.summary)
for i in range(0,len(toi1)):
    edu=TextBlob(toi1[i])
    x=edu.sentiment.polarity
    z=sentiment(toi1[i])
    if x==0:
        neu.append(toi1[i])
        neu_link.append(link4[i])
        neu_img.append(img4[i])
        neu_art.append(art4[i])
        continue
    if z==1:
        pos.append(toi1[i])
        pos_link.append(link4[i])
        pos_img.append(img4[i])
        pos_art.append(art4[i])
        
    elif z==-1:
        neg.append(toi1[i])
        neg_link.append(link4[i])
        neg_img.append(img4[i])
        neg_art.append(art4[i])
        
    
toi1=tuple(zip(toi1,link4,img4,art4))


    
ie_r1 = requests.get("https://indianexpress.com/section/india/")
ie_soup1 = BeautifulSoup(ie_r1.content, 'html5lib')
ie_headings1 = ie_soup1.find_all('h2')
ie_headings1 = ie_headings1[0:4]
ie1=[]
for ph in ie_headings1:
    ie1.append(ph.text.strip())
    headline.append(ph.text.strip())
link5=[]
for l in range(0,len(ie_headings1)): 
    txt=str(ie_headings1[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link5.append(new)

art5=[]
img5=[]
for i in range(0,len(link5)):
    article=Article(link5[i])
    article.download()
    article.parse()
    article.nlp()
    img5.append(article.top_image)
    art5.append(article.summary)
for i in range(0,len(ie1)):
    edu=TextBlob(ie1[i])
    x=edu.sentiment.polarity
    z=sentiment(ie1[i])
    if x==0:
        neu.append(ie1[i])
        neu_link.append(link5[i])
        neu_img.append(img5[i])
        neu_art.append(art5[i])
        continue
    if z==1:
        pos.append(ie1[i])
        pos_link.append(link5[i])
        pos_img.append(img5[i])
        pos_art.append(art5[i])
        
    elif z==-1:
        neg.append(ie1[i])
        neg_link.append(link5[i])
        neg_img.append(img5[i])
        neg_art.append(art5[i])
        
    
ie1=tuple(zip(ie1,link5,img5,art5))


it_r2 = requests.get("https://www.indiatoday.in/movies/bollywood")
it_soup2 = BeautifulSoup(it_r2.content, 'html5lib')
it_headings2 = it_soup2.find_all('h2')
it_headings2 = it_headings2[0:4]
it2=[]
for hm in it_headings2:
    it2.append(hm.text)
    headline.append(hm.text)
link6=[]
for l in range(0,len(it_headings2)): 
    txt=str(it_headings2[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link6.append(web+new)    
   
art6=[]
img6=[]
for i in range(0,len(link6)):
    article=Article(link6[i])
    article.download()
    article.parse()
    article.nlp()
    img6.append(article.top_image)
    art6.append(article.summary)
for i in range(0,len(it2)):
    edu=TextBlob(it2[i])
    x=edu.sentiment.polarity
    z=sentiment(it2[i])
    if x==0:
        neu.append(it2[i])
        neu_link.append(link6[i])
        neu_img.append(img6[i])
        neu_art.append(art6[i])
        continue
    if z==1:
        pos.append(it2[i])
        pos_link.append(link6[i])
        pos_img.append(img6[i])
        pos_art.append(art6[i])
        
    elif z==-1:
        neg.append(it2[i])
        neg_link.append(link6[i])
        neg_img.append(img6[i])
        neg_art.append(art6[i])
        
    
it2=tuple(zip(it2,link6,img6,art6)) 



toi_r2 = requests.get("https://timesofindia.indiatimes.com/briefs/entertainment")
toi_soup2 = BeautifulSoup(toi_r2.content, 'html5lib')
toi_headings2 = toi_soup2.find_all('h2')
toi_headings2 = toi_headings2[2:6]
toi2=[]
for mhm in toi_headings2:
    toi2.append(mhm.text)
    headline.append(mhm.text)
link7=[]
for l in range(0,len(toi_headings2)): 
    txt=str(toi_headings2[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link7.append(web1+new)    

art7=[]
img7=[]
for i in range(0,len(link7)):
    article=Article(link7[i])
    article.download()
    article.parse()
    article.nlp()
    img7.append(article.top_image)
    art7.append(article.summary)
for i in range(0,len(toi2)):
    edu=TextBlob(toi2[i])
    x=edu.sentiment.polarity
    z=sentiment(toi2[i])
    if x==0:
        neu.append(toi2[i])
        neu_link.append(link7[i])
        neu_img.append(img7[i])
        neu_art.append(art7[i])
        continue
    if z==1:
        pos.append(toi2[i])
        pos_link.append(link7[i])
        pos_img.append(img7[i])
        pos_art.append(art7[i])
        
    elif z==-1:
        neg.append(toi2[i])
        neg_link.append(link7[i])
        neg_img.append(img7[i])
        neg_art.append(art7[i])
        
    
toi2=tuple(zip(toi2,link7,img7,art7))    
    

    
ie_r2 = requests.get("https://indianexpress.com/section/lifestyle/")
ie_soup2 = BeautifulSoup(ie_r2.content, 'html5lib')
ie_headings2 = ie_soup2.find_all('h2')
ie_headings2 = ie_headings2[0:4]
ie2=[]
for ph in ie_headings2:
    ie2.append(ph.text.strip())
    headline.append(ph.text.strip())
link8=[]
for l in range(0,len(ie_headings2)): 
    txt=str(ie_headings2[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link8.append(new)    
art8=[]
img8=[]
for i in range(0,len(link8)):
    article=Article(link8[i])
    article.download()
    article.parse()
    article.nlp()
    img8.append(article.top_image)
    art8.append(article.summary)
for i in range(0,len(ie2)):
    edu=TextBlob(ie2[i])
    x=edu.sentiment.polarity
    z=sentiment(ie2[i])
    if x==0:
        neu.append(ie2[i])
        neu_link.append(link8[i])
        neu_img.append(img8[i])
        neu_art.append(art8[i])
        continue
    if z==1:
        pos.append(ie2[i])
        pos_link.append(link8[i])
        pos_img.append(img8[i])
        pos_art.append(art8[i])
        
    elif z==-1:
        neg.append(ie2[i])
        neg_link.append(link8[i])
        neg_img.append(img8[i])
        neg_art.append(art8[i])
        
    
ie2=tuple(zip(ie2,link8,img8,art8))    


it_r3 = requests.get("https://www.indiatoday.in/technology/news")
it_soup3 = BeautifulSoup(it_r3.content, 'html5lib')
it_headings3 = it_soup3.find_all('h2')
it_headings3 = it_headings3[0:4]
it3=[]
for hm in it_headings3:
    it3.append(hm.text)
    headline.append(hm.text)
link9=[]
for l in range(0,len(it_headings3)): 
    txt=str(it_headings3[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link9.append(web+new)    

art9=[]
img9=[]
for i in range(0,len(link9)):
    article=Article(link9[i])
    article.download()
    article.parse()
    article.nlp()
    img9.append(article.top_image)
    art9.append(article.summary)
for i in range(0,len(it3)):
    edu=TextBlob(it3[i])
    x=edu.sentiment.polarity
    z=sentiment(it3[i])
    if x==0:
        neu.append(it3[i])
        neu_link.append(link9[i])
        neu_img.append(img9[i])
        neu_art.append(art9[i])
        continue
    if z==1:
        pos.append(it3[i])
        pos_link.append(link9[i])
        pos_img.append(img9[i])
        pos_art.append(art9[i])
        
    elif z==-1:
        neg.append(it3[i])
        neg_link.append(link9[i])
        neg_img.append(img9[i])
        neg_art.append(art9[i])
        
    
it3=tuple(zip(it3,link9,img9,art9))      
    
  

toi_r3 = requests.get("https://timesofindia.indiatimes.com/briefs/gadgets")
toi_soup3 = BeautifulSoup(toi_r3.content, 'html5lib')
toi_headings3 = toi_soup3.find_all('h2')
toi_headings3 = toi_headings3[2:6]
toi3=[]
for mhm in toi_headings3:
    toi3.append(mhm.text)
    headline.append(mhm.text)
link10=[]
for l in range(0,len(toi_headings3)): 
    txt=str(toi_headings3[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link10.append(web1+new)    
art10=[]
img10=[]
for i in range(0,len(link10)):
    article=Article(link10[i])
    article.download()
    article.parse()
    article.nlp()
    img10.append(article.top_image)
    art10.append(article.summary)
for i in range(0,len(toi3)):
    edu=TextBlob(toi3[i])
    x=edu.sentiment.polarity
    z=sentiment(toi3[i])
    if x==0:
        neu.append(toi3[i])
        neu_link.append(link10[i])
        neu_img.append(img10[i])
        neu_art.append(art10[i])
        continue
    if z==1:
        pos.append(toi3[i])
        pos_link.append(link10[i])
        pos_img.append(img10[i])
        pos_art.append(art10[i])
        
    elif z==-1:
        neg.append(toi3[i])
        neg_link.append(link10[i])
        neg_img.append(img10[i])
        neg_art.append(art10[i])
        
    
toi3=tuple(zip(toi3,link10,img10,art10))  

  
    
ie_r3 = requests.get("https://indianexpress.com/section/technology/gadgets/")
ie_soup3 = BeautifulSoup(ie_r3.content, 'html5lib')
ie_headings3 = ie_soup3.find_all('h2')
ie_headings3 = ie_headings3[0:4]
ie3=[]
for ph in ie_headings3:
    ie3.append(ph.text.strip())
    headline.append(ph.text.strip())
link11=[]
for l in range(0,len(ie_headings3)): 
    txt=str(ie_headings3[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link11.append(new)    
art11=[]
img11=[]
for i in range(0,len(link11)):
    article=Article(link11[i])
    article.download()
    article.parse()
    article.nlp()
    img11.append(article.top_image)
    art11.append(article.summary)
for i in range(0,len(ie3)):
    edu=TextBlob(ie3[i])
    x=edu.sentiment.polarity
    z=sentiment(ie3[i])
    if x==0:
        neu.append(ie3[i])
        neu_link.append(link11[i])
        neu_img.append(img11[i])
        neu_art.append(art11[i])
        continue
    if z==1:
        pos.append(ie3[i])
        pos_link.append(link11[i])
        pos_img.append(img11[i])
        pos_art.append(art11[i])
        
    elif z==-1:
        neg.append(ie3[i])
        neg_link.append(link11[i])
        neg_img.append(img11[i])
        neg_art.append(art11[i])
        
    
ie3=tuple(zip(ie3,link11,img11,art11))    

    
    
it_r4 = requests.get("https://www.indiatoday.in/sports/ipl2020/news")
it_soup4 = BeautifulSoup(it_r4.content, 'html5lib')
it_headings4 = it_soup4.find_all('h2')
it_headings4 = it_headings4[0:4]
it4=[]
for hm in it_headings4:
    it4.append(hm.text)
    headline.append(hm.text)
link12=[]
for l in range(0,len(it_headings4)): 
    txt=str(it_headings4[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link12.append(web+new)    
art12=[]
img12=[]
for i in range(0,len(link12)):
    article=Article(link12[i])
    article.download()
    article.parse()
    article.nlp()
    img12.append(article.top_image)
    art12.append(article.summary)
for i in range(0,len(it4)):
    edu=TextBlob(it4[i])
    x=edu.sentiment.polarity
    z=sentiment(it4[i])
    if x==0:
        neu.append(it4[i])
        neu_link.append(link12[i])
        neu_img.append(img12[i])
        neu_art.append(art12[i])
        continue
    if z==1:
        pos.append(it4[i])
        pos_link.append(link12[i])
        pos_img.append(img12[i])
        pos_art.append(art12[i])
        
    elif z==-1:
        neg.append(it4[i])
        neg_link.append(link12[i])
        neg_img.append(img12[i])
        neg_art.append(art12[i])
        
    

it4=tuple(zip(it4,link12,img12,art12))



toi_r4 = requests.get("https://timesofindia.indiatimes.com/briefs/sports")
toi_soup4 = BeautifulSoup(toi_r4.content, 'html5lib')
toi_headings4 = toi_soup4.find_all('h2')
toi_headings4 = toi_headings4[2:6]
toi4=[]
for mhm in toi_headings4:
    toi4.append(mhm.text)
    headline.append(mhm.text)
link13=[]
for l in range(0,len(toi_headings4)): 
    txt=str(toi_headings4[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link13.append(web1+new)    
art13=[]
img13=[]
for i in range(0,len(link13)):
    article=Article(link13[i])
    article.download()
    article.parse()
    article.nlp()
    img13.append(article.top_image)
    art13.append(article.summary) 
for i in range(0,len(toi4)):
    edu=TextBlob(toi4[i])
    x=edu.sentiment.polarity
    z=sentiment(toi4[i])
    if x==0:
        neu.append(toi4[i])
        neu_link.append(link13[i])
        neu_img.append(img13[i])
        neu_art.append(art13[i])
        continue
    if z==1:
        pos.append(toi4[i])
        pos_link.append(link13[i])
        pos_img.append(img13[i])
        pos_art.append(art13[i])
        
    elif z==-1:
        neg.append(toi4[i])
        neg_link.append(link13[i])
        neg_img.append(img13[i])
        neg_art.append(art13[i])
        
    
toi4=tuple(zip(toi4,link13,img13,art13))   

    

ie_r4 = requests.get("https://indianexpress.com/section/sports/cricket/")
ie_soup4 = BeautifulSoup(ie_r4.content, 'html5lib')
ie_headings4 = ie_soup4.find_all('h2')
ie_headings4 = ie_headings4[0:4]
ie4=[]
for ph in ie_headings4:
    ie4.append(ph.text.strip())
    headline.append(ph.text.strip())
link14=[]
for l in range(0,len(ie_headings4)): 
    txt=str(ie_headings4[l])
    for i in range(0, len(txt)):
        
        if(txt[i]=="a" and txt[i+1]==" " and txt[i+2]=="h" and txt[i+3]=="r" and txt[i+4]=="e" and txt[i+5]=="f" and txt[i+6]=="="):
            new=""
            for j in range(i+8,len(txt)):
                if(txt[j]=='"'):
                    break
                new=new+txt[j]
                j=j+1
            link14.append(new)
art14=[]
img14=[]
for i in range(0,len(link14)):
    article=Article(link14[i])
    article.download()
    article.parse()
    article.nlp()
    img14.append(article.top_image)
    art14.append(article.summary)
for i in range(0,len(ie4)):
    edu=TextBlob(ie4[i])
    x=edu.sentiment.polarity
    z=sentiment(ie4[i])
    if x==0:
        neu.append(ie4[i])
        neu_link.append(link14[i])
        neu_img.append(img14[i])
        neu_art.append(art14[i])
        continue
    if z==1:
        pos.append(ie4[i])
        pos_link.append(link14[i])
        pos_img.append(img14[i])
        pos_art.append(art14[i])
        
    if z==-1:
        neg.append(ie4[i])
        neg_link.append(link14[i])
        neg_img.append(img14[i])
        neg_art.append(art14[i])
        
    
ie4=tuple(zip(ie4,link14,img14,art14))

pos=tuple(zip(pos,pos_link,pos_img,pos_art))
neg=tuple(zip(neg,neg_link,neg_img,neg_art))
neu=tuple(zip(neu,neu_link,neu_img,neu_art))


consumer_key = 'EOxjQUZBztiTRBQRG4SewSp0l'
consumer_secret = 'jo4h3qW9DSwIlGpAZ12pP0xHDa0qePPkKX2uswbpa6xrAQoxAj'
access_token = '1348835624-apgxylphWsCprKdGwG6DFhp0tViFpvet1nEfxua'
access_secret = '9h7BFke7lfI1hLHMv1Rv2NE69NEXaTwy9DBr11wIWZJxa'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

usersData=pd.read_csv('News.csv')


usersData.drop_duplicates(inplace = True)
usersData = usersData[(usersData.totalTweets > 10) & (usersData.Popularity > 1)]
usersData = usersData.reset_index(drop=True)

def getTweets(user):
    twitterUser = api.get_user(user)
    
    tweets = api.user_timeline(screen_name = user, count = 10,tweet_mode='extended')
    tentweets = []
    for tweet in tweets:
        if tweet.full_text.startswith("RT @") == True:
            tentweets.append(tweet.retweeted_status.full_text)
        else:
            tentweets.append(tweet.full_text)
       
    return tentweets
vfunc = np.vectorize(getTweets)
usersData["tweets"] = usersData['ActiveNewsReaders'].apply(lambda x: getTweets(x))

ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 

words = set(nltk.corpus.words.words())


def processing(tweets):
    
    cleanedTweets = []
    for tweet in tweets:
        tw = re.sub('http\S+', '', tweet) 
        tw = re.sub('RT', '', tw) 
        tw = re.sub('@[^\s]+','',tw) 
        tw = "".join([char for char in tw if char not in string.punctuation]) 
        tw = tw.lower() 
        tw = ' '.join([word for word in tw.split() if word not in (stop)])
        tw = ' '.join([word for word in tw.split() if len(word)>2])
        cleanedTweets.append(tw)

    cleanedTweets = ' '.join(cleanedTweets)
    
    
    ProcessedTweets = nltk.word_tokenize(cleanedTweets)
    
    ProcessedTweets = [ps.stem(word) for word in ProcessedTweets]
    
    ProcessedTweets = [wn.lemmatize(word) for word in ProcessedTweets]
    
    ProcessedTweets = [word for word in ProcessedTweets if len(word)>2]
    
    ProcessedTweets = ' '.join(w for w in ProcessedTweets if w in words)
    
    return ProcessedTweets

usersData["ptweets"] = usersData['tweets'].apply(lambda x : processing(x))




tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,min_df=0.1, use_idf=True)

tfidf_matrix = tfidf_vectorizer.fit_transform(usersData.ptweets)
tfidf_matrix.toarray()
pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names(), index = usersData.ActiveNewsReaders)
num_clusters = 5

km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
km.cluster_centers_.argsort()[:, ::-1]
top_keywords=[]
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(num_clusters):
    for ind in order_centroids[i, :10]:
        top_keywords.append(terms[ind])



def processArticles(articles):
    
    cleanedarticles = []
    for article in articles: 
        article = re.sub("[^a-zA-Z]"," ", str(article))
        article = article.lower() 
        article = ' '.join([word for word in article.split() if word not in (stop)])
        article = ' '.join([word for word in article.split() if len(word)>2])   
    
        #tokenization
        article = nltk.word_tokenize(article)
    
        #stemming
        article = [ps.stem(word) for word in article]
    
        #lammitization
        article = [wn.lemmatize(word) for word in article]
    
        article = [word for word in article if len(word)>2]
        article = ' '.join(w for w in article if w in words)
    
        cleanedarticles.append(article)
    return cleanedarticles
headlines = processArticles(headline)
top_keywords = list(dict.fromkeys(top_keywords))
recommended_headlines=[]
for xl in range(0,len(headline)):
    splitting=str(headlines[xl]).split()
    for bq in range(0,len(splitting)):
        for resq in top_keywords:
            if(splitting[bq]==resq):
                recommended_headlines.append(headline[xl])
                
def countele(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 
counting=[]
for xl in range(0,len(recommended_headlines)):
    counting.append(countele(recommended_headlines,recommended_headlines[xl]))
head_count = dict(zip(recommended_headlines,counting)) 

head_count = dict(sorted(head_count.items(), key=operator.itemgetter(1),reverse=True))
recommended_headlines = []
for k in head_count.keys():
    recommended_headlines.append(k) 
recommended_headlines=recommended_headlines[0:6]



dist = 1 - cosine_similarity(tfidf_matrix)
titles = usersData.ActiveNewsReaders
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
posix = mds.fit_transform(dist)
xs, ys = posix[:, 0], posix[:, 1]
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles.values)) 

groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(12, 5)) 
ax.margins(0.05)
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',      
        which='both',    
        bottom='off',  
        top='off',      
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',        
        which='both',     
        left='off',      
        top='off',        
        labelleft='off')
    
ax.legend(numpoints=1) 
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)  

plt.plot()
fig=plt.gcf()
buf=io.BytesIO()
fig.savefig(buf,format='png')
buf.seek(0)
string=base64.b64encode(buf.read())
uri=urllib.parse.quote(string)



def index(req):
    return render(req,'news/index.html',{'w_temp':w_temp, 'w_press':w_press, 'w_hum':w_hum, 'w_desc':w_desc, 'city':city})

def world(req):
    return render(req,'news/world.html',{'it':it, 'toi': toi, 'ie': ie})

def india(req):
    return render(req,'news/india.html',{'it1':it1, 'toi1': toi1, 'ie1': ie1})

def entertainment(req):
    return render(req,'news/entertainment.html',{'it2':it2, 'toi2': toi2, 'ie2': ie2})

def technology(req):
    return render(req,'news/technology.html',{'it3':it3, 'toi3': toi3, 'ie3': ie3})

def sports(req):
    return render(req,'news/sports.html',{'it4':it4, 'toi4': toi4, 'ie4': ie4})

def positive(req):
    return render(req,'news/positive.html',{'pos':pos})
    
def negative(req):
    return render(req,'news/negative.html',{'neg':neg})
    
def neutral(req):
    return render(req,'news/neutral.html',{'neu':neu})
    
def recommend(req):
    return render(req,'news/recommend.html',{'recommended_headlines':recommended_headlines})


def diag(req):
    return render(req, 'news/diag.html',{'data':uri})
