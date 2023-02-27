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
import matplotlib.pyplot as plt
import numpy as np
'''
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold
'''

class predict():
    def __init__(self, df, classifiers, scaler = False, train_size=None, test_size=None) -> None:
        
        
        self.data = df
        
        Y = df['label_model_name']
        
        #X = df.drop(['label_model_name','Unnamed: 0'], axis = 'columns')
        X = df.drop(['label_model_name'], axis = 'columns')
        
        if scaler:
            if scaler == 'minmax':
                scaler = MinMaxScaler()
            elif scaler == 'standard':
                scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else: 
            X = X.to_numpy()
        
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=31, stratify=Y, train_size=train_size, test_size=test_size
        )
        #print(X)
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train.to_numpy()
        self.Y_test = Y_test.to_numpy()
        self.num_classes = len(Y.unique())
        #self.input_size = len(X.columns)
        self.X_train_length=len(self.X_train)

        
        

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

    def get_acc_to_plot_graph_ts_vs_acc(self):
        acc_estimator_wise_li = list()
        for estimator in self.estimators:
            self.predicted = estimator.predict(self.X_test)
            acc = accuracy_score(self.Y_test, self.predicted)
            acc_estimator_wise_li.append(acc)
        return acc_estimator_wise_li
    

def classifier_labeler(classifiers):
    classifiers_label = list()
    for c in classifiers:
        classifiers_label.append(str(c))
    print(classifiers_label)
    return classifiers_label
    
def plot_training_size_var(classifiers,train_size_arr):
    classifiers_label=classifier_labeler(classifiers)

    acc_li = list()
    for train_size in train_size_arr:
        pred = predict(df,classifiers = classifiers,scaler = 'standard', train_size = train_size)    
        pred.estimator_generation()
        acc = pred.get_acc_to_plot_graph_ts_vs_acc()
        acc_li.append(acc)

    acc_np = np.array(acc_li).T
    for ac ,cl in zip(acc_np,classifiers_label):
        num_of_train_data = train_size_arr*pred.X_train_length
        plt.plot(num_of_train_data,ac,label = cl)

    plt.legend()
    plt.title("graph")
    plt.xlabel("train size")
    plt.ylabel("accuracy")
    plt.show()


    

if __name__ == "__main__":
    df = pd.read_csv('./pred_model_trainable_data.csv')
    train_size = 0.33
    classifiers = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
    
    pred = predict(df,classifiers = classifiers,scaler = 'standard', train_size = train_size) # or you can use 'minmax' to use minmax normalization
    #print(pred.Y_test)
    
    pred.estimator_generation()
    pred.predict_all(profile = True)
    

    # plot train_size vs accuracy graph

    train_size_np_arr = np.arange(0.1,0.8,0.05)
    plot_training_size_var(classifiers,train_size_np_arr)

    




