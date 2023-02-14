from sklearn.naive_bayes import GaussianNB
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn.preprocessing import StandardScaler,MinMaxScaler
'''
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold
'''

class predict():
    def __init__(self, df, classifiers, scaler = False, train_size=None, test_size=None) -> None:
        
        
        self.data = df
        
        
        
        Y = df['label_model_name']
        X = df.drop(['label_model_name','Unnamed: 0'], axis = 'columns')
        
        if scaler:
            if scaler == 'minmax':
                scaler = MinMaxScaler()
            elif scaler == 'standard':
                scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else: 
            X = X.to_numpy()
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=41, stratify=Y, train_size=train_size, test_size=test_size
        )
        #print(X)
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train.to_numpy()
        self.Y_test = Y_test.to_numpy()
        self.num_classes = len(Y.unique())
        #self.input_size = len(X.columns)

        
        

        self.classifiers = classifiers
        self.estimators = list()

    def estimator_generation(self):

        for classifier in self.classifiers:
            
            self.estimator = classifier.fit(self.X_train, self.Y_train)
            self.estimators.append(self.estimator)

    def predict_all(self,profile = True):
        #print(self.estimators)
        for estimator in self.estimators:
            
            self.predicted = estimator.predict(self.X_test)
            #print(predicted)
            if profile:
                print('(      Y_test      ,      predicted   )  : count # ')
                pred.profile_wrongly_predicted()
            acc = accuracy_score(self.Y_test, self.predicted)
            
            print('-------------------------------------------------')
            print("{} Accuracy:{}".format(estimator,round(acc,3)))
            
            
            print('#################################################')
            print('#################################################')
    
    def profile_wrongly_predicted(self):
        wrongly_predicted = defaultdict(int)
        for p,y in zip(self.predicted,self.Y_test):
            if p!=y:
                wrongly_predicted[(p,y)] += 1
        if wrongly_predicted:        
            for wrong_set,count in wrongly_predicted.items():
                print(wrong_set,':',count)
        else:
            print('Accuracy is 100%')
    

if __name__ == "__main__":
    df = pd.read_csv('./pred_model_trainable_data.csv')
    train_size = 0.33
    classifiers = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
    pred = predict(df,classifiers = classifiers,scaler = 'standard', train_size = train_size) # or you can use 'minmax' to use minmax normalization
    #print(pred.Y_test)
    
    pred.estimator_generation()
    pred.predict_all(profile = True)

    




