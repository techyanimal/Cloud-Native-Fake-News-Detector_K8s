import joblib
import re
from scipy.sparse import hstack
import numpy as np
from sklearn.decomposition import TruncatedSVD   
import model                                                                          

#loading the model and ohter necessary things
model = joblib.load('Fake_news_predictor.pkl')
tfidf_title = joblib.load('tfidf_title.pkl')
tfidf_text = joblib.load('tfidf_text.pkl')                                                                                                  
def text_processing (text):
    if (isinstance(text,str)):
        text = text.lower()
        text = re.sub(r'http\S+|https\S+|www\S+', '', text)  #removing urls
        text = re.sub(r'<.*?>', '', text)  #removing html tags
        text = re.sub(r'[^\w\s]', '', text)   #removing punctuation marks
        text = re.sub(r'\s+', ' ', text).strip()  #removing extra whitespaces
    else:
        pass
    return text    

#take input                                                                                                                                 
title = input("Enter the title of the news article: ")
text = input("Enter the text of the news article: ")                                                                               
new_title = text_processing(title)
new_text = text_processing(text)
new_title_tfidf = tfidf_title.transform([new_title])
new_text_tfidf = tfidf_text.transform([new_text])                                                                               
new_data_combined = hstack([new_title_tfidf, new_text_tfidf])

# Predict using the loaded model
prediction = model.predict(new_data_combined)

# Output the result
if prediction[0] == 0:
    print("The news is classified as REAL.")
else:
    print("The news is classified as FAKE.")


