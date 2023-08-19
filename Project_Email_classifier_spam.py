#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("C:\\Users\\umar1\\OneDrive\\Documents\\Desktop\\email_spam_classification\\spam2.csv",encoding_errors= 'replace')


# # Data Cleaning

# In[4]:


df


# In[5]:


df.rename(columns={'v1':'spam','v2':'email'},inplace=True)


# In[6]:


df


# In[7]:


df=df.drop({'Unnamed: 2','Unnamed: 3','Unnamed: 4'},axis=1)


# In[8]:


df


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[10]:


df['spam']=encoder.fit_transform(df['spam'])


# In[11]:


df


# In[12]:


df.isnull().sum()
#There are no null values


# In[13]:


df.duplicated().sum()


# In[14]:


df.drop_duplicates(inplace=True)


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# ## Exploratory Data Analysis

# In[17]:


df.head()


# In[18]:


df['spam'].value_counts()


# In[19]:


# Now here we see that the proportion of ham is much more than that of the spam--- data is imbalanced
plt.pie(df['spam'].value_counts(),labels=['ham','spam'])


# In[20]:


import nltk


# In[21]:


nltk.download('punkt')


# In[22]:


#no.of characters present is determined here.
df['num_characters']=df['email'].apply(len)


# In[23]:


df['num_word']=df['email'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[24]:


df


# In[25]:


df['num_sentences']=df['email'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[26]:


df


# In[27]:


df[['num_characters','num_word','num_sentences']].describe()


# In[28]:


df[df['spam']==1][['num_characters','num_word','num_sentences']].describe()


# In[29]:


df[df['spam']==0][['num_characters','num_word','num_sentences']].describe()


# In[30]:


# Now let us visualize the cases of number of words and characters for the two cases of spam and ham
fig, ax = plt.subplots(figsize=(20, 5))
sn.histplot(df[df['spam']==0]['num_characters'],color='red')
sn.histplot(df[df['spam']==1]['num_characters'],color='black')


# In[31]:


fig, ax = plt.subplots(figsize=(20, 5))
sn.histplot(df[df['spam']==0]['num_word'],color='red')
sn.histplot(df[df['spam']==1]['num_word'],color='black')


# In[32]:


sn.pairplot(df,hue='spam')   ## Tells us the relation between each variables.


# In[33]:


sn.heatmap(df.corr(),annot=True)


# In[34]:


#There we can see that the correlation between the coefficient therefore multilinearity is present , Therefore we see that num_characters has the most correlation with spam hence we take that.


# # Text Preprocessing
# 

# In[35]:


nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from string import punctuation
from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[36]:


def transform_text(text):
    text=text.lower()
    
    text=nltk.word_tokenize(text)
    
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


# In[37]:


transform_text("Hi How are You %&&&& i love dancing")


# In[38]:


df['transformed_text']=df['email'].apply(transform_text)


# In[39]:


df


# In[40]:


#Shows the top 30 most frequent values in case of ham messages

spam_corpus=[]
for i in df[df['spam']==0]['transformed_text'].to_list():
    for word in i.split():
      spam_corpus.append(word)  
spam_corpus


# In[41]:


from collections import Counter
sn.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[42]:


#Shows the top 30 most frequent values in case of spam messages
spam_corpus=[]
for i in df[df['spam']==1]['transformed_text'].to_list():
    for word in i.split():
      spam_corpus.append(word)  
spam_corpus


# In[43]:


from collections import Counter
sn.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# # Model Building 

# In[44]:


# now we must do the vectorization of the sentences inorder to apply ml algorithms on them. Therefore we can either use ,
#bag of words
#TF_IDF

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf=TfidfVectorizer(max_features=3000)
cv=CountVectorizer()


# In[45]:


X=tfidf.fit_transform(df['transformed_text']).toarray()


# In[46]:


X.shape


# In[47]:


Y=df['spam'].values


# In[48]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[49]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


# In[50]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[51]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[52]:



gnb.fit(X_train,Y_train)


# In[53]:


Y_pred=gnb.predict(X_test)


# In[54]:


for i in [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision_score(Y_test,Y_pred)]:
    print(i)


# In[55]:



mnb.fit(X_train,Y_train)
Y_pred=mnb.predict(X_test)
for i in [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision_score(Y_test,Y_pred)]:
    print(i)


# In[56]:


bnb.fit(X_train,Y_train)
Y_pred=bnb.predict(X_test)
for i in [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred),precision_score(Y_test,Y_pred)]:
    print(i)


# We can see that in this case the multinomial naive bayes gives the precision of 1 and accuracy is also higher than the other two naive bayes

# # Model building followed by Bag of words

# In[57]:


X1=cv.fit_transform(df['transformed_text']).toarray()


# In[58]:


Y1=df['spam']
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2)


# In[59]:


gnb.fit(X1_train,Y_train)
Y1_pred=gnb.predict(X1_test)
print(accuracy_score(Y1_test,Y1_pred))
print(confusion_matrix(Y1_test,Y1_pred))
print(precision_score(Y1_test,Y1_pred))


# In[60]:


mnb.fit(X1_train,Y_train)
Y1_pred=mnb.predict(X1_test)
print(accuracy_score(Y1_test,Y1_pred))
print(confusion_matrix(Y1_test,Y1_pred))
print(precision_score(Y1_test,Y1_pred))


# In[61]:


bnb.fit(X1_train,Y_train)
Y1_pred=bnb.predict(X1_test)
print(accuracy_score(Y1_test,Y1_pred))
print(confusion_matrix(Y1_test,Y1_pred))
print(precision_score(Y1_test,Y1_pred))


# Therefore we see that the vectorization by TFIDF gives a better result than the bag of words, Hence we use a model of TF_IDF followed by multinomialNaiveBayes

# In[62]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[63]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[64]:


#lets make a dictionary of the algorithms
clfs={
    'SVC':svc,
    'KNC':knc,
    'MNB':mnb,
    'DTC':dtc,
    'LRC':lrc,
    'RFC':rfc,
    'ABC':abc,
    'BC':bc,
    'ETC':etc,
    'GBDT':gbdt
}


# In[65]:


def training_data (clf,X_train,X_test,Y_train,Y_test):
    clf.fit(X_train,Y_train)
    Y_predicted=clf.predict(X_test)
    acc_score=accuracy_score(Y_test,Y_predicted)
    prec_score=precision_score(Y_test,Y_predicted)
    return acc_score,prec_score


# In[66]:


accuracy=[]
precision=[]
for name,clf in clfs.items():
    current_accuracy,current_precision = training_data(clf, X_train,X_test,Y_train,Y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy.append(current_accuracy)
    precision.append(current_precision)


# In[67]:


performance_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy,'Precision':precision}).sort_values('Precision',ascending=False)


# In[68]:


performance_df


# In[69]:


performance_df_1=pd.melt(performance_df,id_vars=['Algorithm'])


# In[70]:


performance_df_1


# In[71]:


sn.catplot(x='Algorithm',y='value',hue='variable',data=performance_df_1,kind='bar',height=8)
plt.ylim(0.5,1)


# ### THEREFORE WE SEE THAT THE MULTINOMIAL NAIVE BAYES GIVES THE BEST RESULT BASED ON PRECISION AND ALSO ACCURACY WHENEVER WE ARE USING IT AFTER VECTORIZING WITH TF-IDF

# In[72]:


#we see that scaling does not make much of a difference on the accuracy and precision , Therefore We go ahead with this model
#with about 3000 max_features


# In[74]:


import pickle
pickle.dump(tfidf,open('new_vectorizer.pkl','wb'))
pickle.dump(mnb,open('model_email.pkl','wb'))


# In[ ]:





# In[ ]:




