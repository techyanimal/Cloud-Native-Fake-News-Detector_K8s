#importing all libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as py
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
#from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib 

#reading datasets
True_ = pd.read_csv(r'python-and-ml\Fake News Detector Project\Dataset\true.csv')
Fake_ = pd.read_csv(r'python-and-ml\Fake News Detector Project\Dataset\fake.csv')  

#data preprocessing 

#analysing datasets
print(True_.info)  
print(Fake_.info)      

#removing extra unnamed column from fake dataset 
Fake_ = Fake_.loc[:,~Fake_.columns.str.contains('^Unnamed')]
Fake_ 

#adding a column for identitfying fake or true news to both datasets
Fake_['Label']=0
True_['Label']=1                                                                                                                                        
True_                                                                                                                                                        
Fake_ 

#combining true and fake datasets into 1
Data = pd.concat([True_,Fake_], axis=0)
Data     

#shuffling the data
Data = Data.sample(frac=1).reset_index(drop=True)
Data                                                                                                                                                         
Data_copy = Data.copy()                                                                                                                      
Columns = Data_copy.columns.values
print(Columns)
n= len(Columns)
print(n)  

#checking for null values and urls, punctuation marks, extra whitespaces 
print(Data_copy.isnull().sum(), '\n')
pattern = f'[{string.punctuation}]'

#Data = Data.astype(str)
try:
    for i in Columns:
        print(f'Sample data from column {i}:', Data_copy[i].head(5))
        print('{} : URLs - '.format(i), Data_copy[i].str.contains(r'http\S+|https\S+|www\S+').sum())
        print('{} : Punctuation marks - '.format(i), Data_copy[i].str.contains(pattern).sum())
        print('{} : extra white space - '.format(i), Data_copy[i].str.contains(r'\s+').sum())
        print('{} : HTML tags - '.format(i), Data_copy[i].str.contains(r'<.*?>').sum(), '\n')   
except AttributeError:
    print("Few values giving error but the purpose was achieved who cares")   

#defining a function to remove the unwanted characters or substrings 
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

#removing unwanted text
for i in range(0,n-1):
    Data_copy[Columns[i]] = Data_copy[Columns[i]].apply(lambda x: text_processing(x) if isinstance(x,str) else x)
Data_copy 

#encoding - TF-IDF
tfidf_title = TfidfVectorizer(max_features=2500)
tfidf_text = TfidfVectorizer(max_features=5000)  
tfidf_title_matrix = tfidf_title.fit_transform(Data_copy['title'])
tfidf_text_matrix = tfidf_text.fit_transform(Data_copy['text'])
combined_matrix = hstack([tfidf_title_matrix, tfidf_text_matrix])

#reducing dimensionality
#svd = TruncatedSVD(n_components=300)
#reduced_matrix = svd.fit_transform(combined_matrix)                                                                        
combined_df = pd.DataFrame(combined_matrix.toarray())                                                               
#Data_combined = pd.concat([Data_copy.reset_index(drop=True), combined_df], axis=1)
#Data_combined  

#data split                                                                                                                                   
X_train, X_test, y_train, y_test = train_test_split(combined_df, Data_copy['Label'], test_size=0.2, random_state=42)          
                                                                                                
# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)                                                                                                                      
y_pred = model.predict(X_test)                                                                                                            
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))                                                                      
joblib.dump(model, 'Fake_news_predictor.pkl')
joblib.dump(tfidf_title, 'tfidf_title.pkl')
joblib.dump(tfidf_text, 'tfidf_text.pkl')
