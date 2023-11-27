#!/usr/bin/env python
# coding: utf-8

# # Variable Information
# * Name: The reviewer's name, if available.
# * Location: The location or city associated with the reviewer, if provided.
# * Date: The date when the review was posted.
# * Rating: The star rating given by the reviewer, ranges from 1 to 5.
# * Review: The textual content of the review, captures the reviewer's experience and opinions.
# * Image Links: Links to images associated with the reviews, if available.


# Importing basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



# Importing libraries for nlp preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


# import sentiment Analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


# import libreries for Vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# Data Scaling Libreries
from sklearn.preprocessing import StandardScaler, Normalizer



# Models Libreries
from sklearn.model_selection import train_test_split

# evaluation matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Random Forest Model 
from sklearn.ensemble import RandomForestClassifier

# Model saving library
import joblib


# Fetching the data.
data = pd.read_csv(r'reviews_data.csv')


# #### Preprocessing

# Text cleaning and tokenization
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ' '.join([word for word in word_tokenize(text) if word.isalnum()])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

data['cleaned_text_review'] = data['Review'].apply(preprocess_text)


# custom features
data['word_count'] = data['cleaned_text_review'].apply(lambda x:len(x.split()))
data['char_count'] = data['cleaned_text_review'].apply(len)


# Combine all documents into a single text
corpus = ' '.join(data['cleaned_text_review'])



sid = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each document
data['Sentiment_score'] = data['cleaned_text_review'].apply(lambda x: sid.polarity_scores(x)['compound'])



def senti(x):
    if x >= 4:
        return 'Positive'
    elif x <= 2:
        return 'Negative'
    else:
        return 'Neutral'


data['Rating_type'] = data.Rating.apply(senti)


# dropping the No Review Text
l = list(data[data.Review == 'No Review Text'].index)
data.drop(index = l,inplace = True)



# after analysing the reviews of Null rating we get that all the reviews are negative reviews. So we give 1 rating to all the Nan values here.

data['Rating'] = np.where(data.Rating.isnull(),1,data.Rating)




# ## Vectorization Techniques

# #### 1. BOW (Bag Of Words)
cv = CountVectorizer()
cv_matrix = cv.fit_transform(data['cleaned_text_review']).toarray()
cv_features = cv.get_feature_names_out()

# print("\n\nBag of Words Matrix:")
# print(cv_matrix)
# print("\n Features Names:")
# print(cv_features)


# #### 2. N-grams

# ##### Bigram
ngram_cv = CountVectorizer(ngram_range=(2,2))
ngram_cv_matrix = ngram_cv.fit_transform(data['cleaned_text_review']).toarray()
ngram_cv_features = ngram_cv.get_feature_names_out()

# print("\n\n\n2-gram Matrix:")
# print(ngram_cv_matrix)
# print("\n Features Names:")
# print(ngram_cv_features)

# ##### Trigram
ngram3_cv = CountVectorizer(ngram_range=(3,3))
ngram3_cv_matrix = ngram3_cv.fit_transform(data['cleaned_text_review']).toarray()
ngram3_cv_features = ngram3_cv.get_feature_names_out()

# print("\n\n\n3-gram Matrix:")
# print(ngram3_cv_matrix)
# print("\n Features Names:")
# print(ngram3_cv_features)


# ##### Quad-gram
ngram4_cv = CountVectorizer(ngram_range=(2,3))
ngram4_cv_matrix = ngram4_cv.fit_transform(data['cleaned_text_review']).toarray()
ngram4_cv_features = ngram4_cv.get_feature_names_out()

# print("\n\n\n4-gram Matrix:")
# print(ngram4_cv_matrix)
# print("\n Features Names:")
# print(ngram4_cv_features)

# #### 3. TFIDF 
tfidf  = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['cleaned_text_review']).toarray()
features_names = tfidf.get_feature_names_out()

# print("\n\n\nTF-IDF Matrix:")
# print(tfidf_matrix)
# print("\n Features Names:")
# print(features_names)


# Prepairing data frame for machine learning model
ready_data = pd.DataFrame(data = tfidf_matrix,columns = features_names)

# Correcting the index of data
data.reset_index(inplace = True)
data.drop(columns='index',inplace = True)

# Adding the word count and char count columns to ready_data
ready_data['word_count']=data['word_count']
ready_data['char_count'] = data['char_count']

# Correcting the Rating feature 
data.Rating = data.Rating.astype('int')
data.Rating = data.Rating-1



# #### Standard Scaler
# Performing on TFIDF data
st = StandardScaler()
st_data = st.fit_transform(ready_data)

st_data = pd.DataFrame(st_data,columns= ready_data.columns)


# #### Normalizer
nor = Normalizer()
nor_data = nor.fit_transform(st_data)

nor_ready_data = pd.DataFrame(nor_data, columns= st_data.columns)


# ### Modeling
# building model on TFIDF data
x = nor_ready_data
y = data.Rating

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=2)


# #### Random Forest

# Building model
rf = RandomForestClassifier(random_state=0)
rf_model = rf.fit(xtrain,ytrain)
pred = rf_model.predict(xtest)

# Calculating for performance of the model.
accuracy = accuracy_score(ytest,pred)
precision = precision_score(ytest,pred,average='weighted')
recall = recall_score(ytest,pred,average='weighted')
f1 = f1_score(ytest,pred,average='weighted')

# Checking for performance of the model.
print('\n\n\nAccuracy score:',round(accuracy,2))
print('F1 score:',round(f1,2))
print('Precision score:',round(precision,2))
print('Recall Score:',round(recall,2))


joblib.dump(rf_model,'Random_Forest_trained.joblib')
joblib.dump(tfidf,'tfidf.joblib')
joblib.dump(st,'scaler.joblib')
joblib.dump(nor,'normalizer.joblib')
# joblib.dump(preprocess_text,'preprocess.joblib')