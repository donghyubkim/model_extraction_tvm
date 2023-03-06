
import pandas as pd 
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class predictBase():
    def __init__(self, df, normalization = 'standard') -> None:
        
        self.data = df
        self.Y = df['label_model_name']
        #X = df.drop(['label_model_name','Unnamed: 0'], axis = 'columns')
        self.X_primary = df.drop(['label_model_name'], axis = 'columns')
        
        if normalization:
            if normalization == 'minmax':
                normalization = MinMaxScaler()
            elif normalization == 'standard':
                normalization = StandardScaler()
            
            self.X = normalization.fit_transform(self.X_primary)
        else: 
            self.X = self.X_primary.to_numpy()
        self.classifier = None
        
        
       
    
    def splitTrainTest(self,train_size):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, random_state=31, train_size=train_size
        )
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train.to_numpy()
        self.Y_test = Y_test.to_numpy()
        self.num_classes = len(self.Y.unique())
        #self.input_size = len(X.columns)
        self.X_train_length=len(self.X_train)
        
    def select_k_best(self,k,score_func):
        selector = SelectKBest(score_func = score_func, k=k)
        print(self.X)
        self.X = selector.fit_transform(self.X, self.Y)
        print(self.X)
        filter =selector.get_support(indices=True)
        
        features = self.X_primary
        
        print(str(score_func))
        print(features.iloc[0,filter])

        features = np.array(self.X_primary)
        print(str(score_func))
        print(features[filter])
        return self.X
        


    def estimatorGeneration(self): 
        
        self.estimator = self.classifier.fit(self.X_train, self.Y_train)
        self.importances = self.estimator.feature_importances_

    def estimatorGeneration_RFE(self,feature_to_select):
        
        classifier = RandomForestClassifier()
        estimator = RFE(estimator=classifier, n_features_to_select=feature_to_select)
        estimator.fit(self.X_train, self.Y_train)
        #print(estimator.support_)
        
        #print('###RFE filtered features###')
        #print(self.X_primary.loc[0,estimator.support_])

        featureRankingDic = dict()

        for rank, feature in zip(estimator.ranking_,list(self.X_primary.columns)):
            featureRankingDic[feature] = rank
        print('###importance ranking###')
        sorted_FRD = sorted(featureRankingDic.items(), key=lambda x: x[1])
        #print(sorted_FRD)
        for key,val in sorted_FRD:
            print(key + ":",val)
        

        
        Y = list(featureRankingDic.values())
        X = np.arange(len(Y))
        X_label = list(featureRankingDic.keys()) 
        plt.plot(X, Y,'go')
        plt.xticks(X,X_label,rotation = 90 , fontsize = 7)

        # Set the x-label
        plt.xlabel('feature')

        # Set the y-label
        plt.ylabel('ranking')

        # Set the plot title
        plt.title('feature importance ranking')

        # Display the plot
        plt.tight_layout()
        plt.show()
        plt.savefig('log/feature_ranking.pdf')




        return estimator.support_, estimator.ranking_


    def predict(self):
        
        self.predicted = self.estimator.predict(self.X_test)
        self.acc = accuracy_score(self.Y_test, self.predicted)
        print(self.acc)
        

    def wronglyPredicted(self):
        
        wronglyPredicted = defaultdict(int)
        for predicted,label in zip(self.predicted,self.Y_test):
            if predicted != label:
                wronglyPredicted[(predicted,label)] += 1
        
        if wronglyPredicted: 
            
            Y = list(wronglyPredicted.values())
            X = np.arange(len(Y))
            X_label = list(wronglyPredicted.keys()) 
            plt.figure(figsize=(6,8))
            plt.bar(X,Y)
            plt.xticks(X,X_label,rotation = 90 , fontsize = 7)
            plt.title(str(self.estimator)+": wrongly predicted")
            plt.xlabel("predicted : answer")
            plt.ylabel("count")
            #plt.show()
            plt.tight_layout()
            plt.savefig('log/Wrongly_predicted/{}.pdf'.format(str(self.classifier).split('(')[0]))
        else:
            pass
    
    def returnAccVsTraningSize(self):
        
        return self.acc,self.X_train_length
    
    def makeClassifierInstance(self,df):
        NB = gaussianNB(df)
        LR = logisticRegression(df)
        RF = randomForest(df)
        MLP = multiLayeredPerceptron(df)
        NC = nearestCentroid(df)
        KNN = kNearestNeighbors(df)
        ADB = adaboost(df)
        classifier_set = {NB,LR,RF,MLP,NC,KNN,ADB}

        return classifier_set
    
    def correlation(self):
        table = self.data
        df = table.pivot('aggregated_duration_nnconv2d', 'aggregated_percentage_nnconv2d', 'labe_model_name')
        sns.heatmap(table)



class gaussianNB(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = GaussianNB()
class logisticRegression(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = LogisticRegression()   
class randomForest(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = RandomForestClassifier()   
class multiLayeredPerceptron(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = MLPClassifier()   
class nearestCentroid(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = NearestCentroid()
class kNearestNeighbors(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = KNeighborsClassifier()
class adaboost(predictBase):
    def __init__(self,df, normalization = 'standard'):
        super().__init__(df, normalization = normalization)
        self.classifier = AdaBoostClassifier()

    

if __name__ == "__main__":
    
    
    df = pd.read_csv('./pred_model_trainable_data.csv')
    train_size = 0.3
    '''
    p=predictBase(df,normalization='minmax')
    classifier_set = p.makeClassifierInstance(df=df)
    score_func_li = [chi2,mutual_info_classif]

    for score_func in score_func_li:
        p.select_k_best(2,score_func=score_func)
    '''

    p = predictBase(df)
    p.splitTrainTest(train_size)
    p.estimatorGeneration_RFE(1)
    '''
    for classifier in classifier_set:
        classifier.splitTrainTest(train_size = train_size)
        #classifier.estimatorGeneration()
        classifier.estimatorGeneration_RFE(3)
        classifier.predict()
        classifier.wronglyPredicted()
    '''
    '''
    trainSizeArray = np.arange(0.1,0.8,0.05)
    columns = ['aggregated_duration_nnconv2d','aggregated_percentage_nnconv2d','aggregated_count_nnconv2d','total_duration','total_percentage','total_count','aggregated_duration_nnbiasadd','aggregated_percentage_nnbiasadd','aggregated_count_nnbiasadd','aggregated_duration_nnrelu','aggregated_percentage_nnrelu','aggregated_count_nnrelu','aggregated_duration_nnmaxpool2d','aggregated_percentage_nnmaxpool2d','aggregated_count_nnmaxpool2d','aggregated_duration_concatenate','aggregated_percentage_concatenate','aggregated_count_concatenate','aggregated_duration_nnglobalavgpool2d','aggregated_percentage_nnglobalavgpool2d','aggregated_count_nnglobalavgpool2d','aggregated_duration_nnavgpool2d','aggregated_percentage_nnavgpool2d','aggregated_count_nnavgpool2d','aggregated_duration_nndense','aggregated_percentage_nndense','aggregated_count_nndense','aggregated_duration_add','aggregated_percentage_add','aggregated_count_add','aggregated_duration_mean','aggregated_percentage_mean','aggregated_count_mean','aggregated_duration_transpose','aggregated_percentage_transpose','aggregated_count_transpose','aggregated_duration_stridedslice','aggregated_percentage_stridedslice','aggregated_count_stridedslice','aggregated_duration_clip','aggregated_percentage_clip','aggregated_count_clip','aggregated_duration_multiply','aggregated_percentage_multiply','aggregated_count_multiply','aggregated_duration_rsqrt','aggregated_percentage_rsqrt','aggregated_count_rsqrt','aggregated_duration_negative','aggregated_percentage_negative','aggregated_count_negative','aggregated_duration_nnpad','aggregated_percentage_nnpad','aggregated_count_nnpad','aggregated_duration_nnlrn','aggregated_percentage_nnlrn','aggregated_count_nnlrn','aggregated_duration_nnsoftmax','aggregated_percentage_nnsoftmax','aggregated_count_nnsoftmax']
    df = pd.read_csv('./pred_model_trainable_data.csv')
    train_size = 0.3
    p=predictBase(df,normalization='minmax')
    p.correlation(columns)
    '''


    #AB = adaboost(df,train_size = train_size)
    #AB.wronglyPredictedLog()
    #NB.predict()
    #NB.wronglyPredictedLog()
    #RF = randomForest(df)
    #classifier = RF
    '''
    NB = randomForest(df)
    classifier = NB
    classifier.splitTrainTest(train_size = train_size)
    classifier.estimatorGeneration()
    classifier.estimatorGeneration_RFE(feature_to_select=3)
    classifier.predict()
    '''
    