from sklearn.naive_bayes import GaussianNB
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('pred_model_trainable_result.csv')
#print(data['label_model_name'])

Y = df['label_model_name']
X = df.drop('label_model_name', axis = 'columns')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=321)

#print(X_test)
#print(X_train)
print(Y_test)
#print(Y_train)
X_test = np.array(X_test)
X_train = np.array(X_train)
Y_test = np.array(Y_test)
Y_train = np.array(Y_train)



gnb = GaussianNB()
print('predicted is')
y_pred = gnb.fit(X_train, Y_train).predict(X_test)
print(y_pred)
#print("Number of mislabeled points out of a total %d points : %d"(X_test.shape[0], (Y_test != y_pred).sum()))