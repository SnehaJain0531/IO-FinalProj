#Accuracy 83.3%
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

df= pd.read_csv('dataset.csv')
df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)

df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))   
df = pd.get_dummies(df, columns=categorical_columns)
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

X = df.loc[:,temp_data.columns!='RainTomorrow']
Y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(X, Y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)]) 
df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
X = df[['RainToday','Humidity3pm']] 
Y = df[['RainTomorrow']]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
print('the length of X_train is:',len(X_train))
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(X_train,Y_train)
Y_pred = dt_clf.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)
print('Accuracy :',accuracy)
