# Importing the libraries
import numpy as np
import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')
reviews=load_files(os.path.join(BASE_DIR, 'txt_sentoken')
X,y=reviews.data,reviews.target
#nltk==natural lamguage tool kit
#stopwords==is,a,an,the

#bag of words model
#NLTK PREPROCESSING
'''
re=regular expression
re.sub(expression,new,var)
\W return a match where there is non-string char
\s return when there is white space char
lower==converts each char in lower
^matches beginning of string
'''
corpus=[]
for i in range(len(X)):
    review=re.sub(r'\W',' ',str(X[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+',' ',review)
    review=re.sub(r's+',' ',review)
    corpus.append(review)
    
#w,r,w+,r+
    
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
with open('X.pickle','rb') as f:
    X=pickle.load(f)
with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
#count vectorizer
    
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(max_features=len(X),min_df=3,max_df=0.6,stop_words=stopwords.words('English'))
  
x=count.fit_transform(corpus).toarray()  

from sklearn.feature_extraction.text import TfidfVectorizer
transformer=TfidfVectorizer()
X=transformer.fit_transform(corpus)
X=X.toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

with open('Classfier.pickle','wb') as f:
    pickle.dump(clf,f)
with open('TFidf.pickle','wb') as f:
    pickle.dump(count,f)

#CONFUSION MATRIX
    
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
sample=['this is a good guy,have a great life']
#WORKING CODE
