
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
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
rc('figure', **{'figsize': (5, 4)})
class predictBase():
    def __init__(self, df, normalization = 'standard') -> None:
        
        self.data = df
        self.Y = df['label_model_name']
        #X = df.drop(['label_model_name','Unnamed: 0'], axis = 'columns')
        self.X_primary = df.drop(['label_model_name'], axis = 'columns')
        
        if normalization:
            if normalization == 'minmax':
                self.normalization = MinMaxScaler()
            elif normalization == 'standard':
                self.normalization = StandardScaler()
            
            self.X = self.normalization.fit_transform(self.X_primary)
        else: 
            self.X = self.X_primary.to_numpy()
        self.classifier = None
        
        
       
    
    def splitTrainTest(self,train_size):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.Y, random_state=5, train_size=train_size
        )
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train.to_numpy()
        self.Y_test = Y_test.to_numpy()
        self.num_classes = len(self.Y.unique())
        #self.input_size = len(X.columns)
        self.X_train_length=len(self.X_train)
        



    def estimatorGeneration(self): 
        
        self.estimator = self.classifier.fit(self.X_train, self.Y_train)
        
    


    def plot_feature_ranking(self,feature_to_select,plot = False):
        
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
        sorted_feature_ranking = sorted(featureRankingDic.items(), key=lambda x: x[1])
        #print(sorted_FRD)
        for key,val in sorted_feature_ranking:
            print(key + ":",val)
        

        if plot:
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

        return estimator.support_, estimator.ranking_, sorted_feature_ranking


    def predict(self):
        
        self.predicted = self.estimator.predict(self.X_test)
        self.acc = accuracy_score(self.Y_test, self.predicted)
        print(self.acc)
        return self.acc
    
    def predict_victim(self,victim_df):
        victim_df = victim_df.drop(['label_model_name'], axis = 'columns')
        self.predicted = self.estimator.predict(self.X_test)
        self.acc = accuracy_score(self.Y_test, self.predicted)
        print(self.acc)
        return self.acc
        

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
    

    def correlation(self):
        table = self.data
        df = table.pivot('aggregated_duration_nnconv2d', 'aggregated_percentage_nnconv2d', 'labe_model_name')
        sns.heatmap(table)

    def binning(self):
        print(self.X_primary)

        self.X_primary = self.X_primary.drop(['total_duration','total_count','total_percentage'], axis = 'columns')
        def binning_internal(row):
            group1 = group2 = group3 = group4 = 0
            for col in row.index:
                if 'conv2d' in col:
                    if 'duration' in col:
                        group1 += row[col]
                    elif 'count' in col:
                        group2 += row[col]
                else:
                    if 'duration' in col:
                        group3 += row[col]
                    elif 'count' in col:
                        group4 += row[col]
            return pd.Series({'group1': group1, 'group2': group2, 'group3': group3, 'group4': group4})

        # apply the binning function to each row
        self.X = pd.DataFrame({})
        self.X[['convolution_duration', 'convolution_count', 'non_convolution_duration', 'non_convolution_count']] = self.X_primary.apply(binning_internal, axis=1)
        print(self.X)
        self.normalization = StandardScaler()
            
        self.X = self.normalization.fit_transform(self.X)
        
            
        print(self.X)


    def plot_learning_curve(self,acc_text = True):
        """
        Plots the learning curve for a list of classifier objects using scikit-learn's learning_curve function.
        """
        

        estimators = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
        #plt.figure(figsize=(10, 6))
        clfname_dict = {'LogisticRegression': 'lr', 'MLPClassifier': 'nn', 'KNeighborsClassifier': 'knn', 'NearestCentroid': 'centroid', 'GaussianNB': 'nb', 'RandomForestClassifier': 'rf', 'AdaBoostClassifier': 'ab'}
        #plt.title("Train with binned 4 features")
        plt.title("")
        plt.xlabel("Number of Profiles per Architecture in Training Dataset")
        plt.ylabel("Architecture Prediction Accuracy")

        for estimator in estimators:
            train_sizes, train_scores, test_scores = learning_curve(estimator, self.X, self.Y, n_jobs=-1, train_sizes=np.linspace(1/20, 1, 20),shuffle=True)
            #train_scores_mean = np.mean(train_scores, axis=1)
            train_sizes = np.arange(1,21)
            test_scores_mean = np.mean(test_scores, axis=1)
            print(train_sizes)
            plt.plot(train_sizes, test_scores_mean, label=clfname_dict[type(estimator).__name__])
            
            '''
            if acc_text:
                for i, accuracy in enumerate(np.mean(test_scores, axis=1)):
                    plt.text(train_sizes[i], accuracy, '{:.2f}'.format(accuracy),fontdict=fontdict)
            '''
        
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("learning_curve_4", dpi=500, bbox_inches="tight")
        plt.show()

   
    def plot_classifiers_accuracy(self,column_sets):

        
        classifiers = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
        #plt.figure(figsize=(10,6))
        data = self.data
        clfname_dict = {'LogisticRegression': 'lr', 'MLPClassifier': 'nn', 'KNeighborsClassifier': 'knn', 'NearestCentroid': 'centroid', 'GaussianNB': 'nb', 'RandomForestClassifier': 'rf', 'AdaBoostClassifier': 'ab'}
        for clf in classifiers:
            accuracies = []
            for cols in column_sets:
                X_train, X_test, y_train, y_test = train_test_split(data[cols], data['label_model_name'], train_size=0.5)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
            plt.plot([len(cols) for cols in column_sets], accuracies, label=clfname_dict[clf.__class__.__name__])
        print(clf.__class__.__name__)
        plt.xlabel('Number of Features to Train Architecture Prediction Model')
        plt.xticks([1,4,7,10,13,16,19,22,25,28])
        plt.ylabel('Architecture Prediction Accuracy')
        plt.title('')
        plt.legend(loc = "lower right")
        plt.tight_layout()
        plt.savefig("Accuracy_num_of_feature", dpi=500, bbox_inches="tight")
        plt.show()
    
    def mitigation(self):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv('resnet18_dummy.csv')
        # Read the CSV file into a pandas dataframe
        df = df.reindex(columns=self.data.columns, fill_value=0)
        df.to_csv('dummy_inserted_reindexed.csv')
        

        # Split the dataframe into different variables based on the values in the "label_model_name" column
        dummy0 = df[df['label_model_name'] == 'resnet18_dummy0.onnx']
        dummy1 = df[df['label_model_name'] == 'resnet18_dummy1.onnx']
        dummy2 = df[df['label_model_name'] == 'resnet18_dummy2.onnx']
        dummy3 = df[df['label_model_name'] == 'resnet18_dummy3.onnx']
        dummy4 = df[df['label_model_name'] == 'resnet18_dummy4.onnx']
        dummy5 = df[df['label_model_name'] == 'resnet18_dummy5.onnx']
        dummy6 = df[df['label_model_name'] == 'resnet18_dummy6.onnx']
        dummy7 = df[df['label_model_name'] == 'resnet18_dummy7.onnx']
        dummy8 = df[df['label_model_name'] == 'resnet18_dummy8.onnx']
        dummy9 = df[df['label_model_name'] == 'resnet18_dummy9.onnx']
        dummy10 = df[df['label_model_name'] == 'resnet18_dummy10.onnx']
        dummy11 = df[df['label_model_name'] == 'resnet18_dummy11.onnx']
        dummy12 = df[df['label_model_name'] == 'resnet18_dummy12.onnx']
        dummy13 = df[df['label_model_name'] == 'resnet18_dummy13.onnx']
        dummy14 = df[df['label_model_name'] == 'resnet18_dummy14.onnx']
        dummy15 = df[df['label_model_name'] == 'resnet18_dummy15.onnx']

        dummy_list = [dummy0,dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7,dummy8,dummy9,dummy10,dummy11,dummy12,dummy13,dummy14,dummy15]
        

        for data in dummy_list:
            data['label_model_name'] = 'resnet18.onnx' #we have to do it here. so we can split various dummy number
        
        self.dummy_list = dummy_list
        plt.figure(figsize=(10,6))
        estimators = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
        for estimator in estimators:
            
            estimator.fit(self.X_train, self.Y_train)
            print(estimator)
            accuracies = []
            for dummy in dummy_list:
                
                Y_dummy = dummy['label_model_name'].to_numpy()
                X_dummy = dummy.drop(['label_model_name'], axis = 'columns').to_numpy()
                
                #X_dummy = self.normalization.fit_transform(X_dummy)

                predicted = estimator.predict(X_dummy)
                print(predicted)
                acc = accuracy_score(Y_dummy, predicted)
                print(acc)
                accuracies.append(acc)
            plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], accuracies, label=estimator.__class__.__name__)
        plt.xlabel('Number of dummy conv2')
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        plt.ylabel('Accuracy')
        plt.title('Accuracy of different estimators with different number of dummies')
        plt.legend()
        plt.show()
        return dummy_list
    def class_10(self):
        df = pd.read_csv('class_10.csv')

        
        plt.figure(figsize=(10,6))
        estimators = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
        for estimator in estimators:
            
            estimator.fit(self.X_train, self.Y_train)
            print(estimator)
            accuracies = []
            Y = df['label_model_name'].to_numpy()
            X = df.drop(['label_model_name'], axis = 'columns').to_numpy()
            
            #X_dummy = self.normalization.fit_transform(X_dummy)
            predicted = estimator.predict(X)
            print(predicted)
            acc = accuracy_score(Y, predicted)
            print(acc)

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
    
    run = 'number of feature vs accuracy'
    df = pd.read_csv('./golden/optlevel0/pred_model_trainable_data.csv')
    
    train_size = 0.5
    if run == 'predict_victim':
        victim_df = pd.read_csv('./aggregated_results/result_victim.csv')
        p = predictBase(df)
        p.splitTrainTest(train_size)
    if run == 'wrongly_predicted':
        LR =logisticRegression(df)
        AB = adaboost(df)
        MLP = multiLayeredPerceptron(df)
        
        e = MLP
        e.splitTrainTest(train_size=train_size)
        e.estimatorGeneration()
        e.predict()
        e.wronglyPredicted()
    if run == 'mitigation':
        p = predictBase(df,normalization = None)
        p.splitTrainTest(train_size)
        p.mitigation()
    if run == 'learning curve':
        p = predictBase(df)
        #p.splitTrainTest(train_size)
        p.plot_learning_curve()
        #p.estimatorGeneration_RFE(1)

    if run == 'learning curve binning':
        p = predictBase(df)
        #p.splitTrainTest(train_size)
        p.binning()
        p.plot_learning_curve()
        
    
    if run == 'feature print':
        print(df.columns)
        print(len(df.columns))

    
    if run == 'number of feature vs accuracy':
        p = predictBase(df)
        p.splitTrainTest(train_size= 0.5)
        _,_,feature_ranking_dict = p.plot_feature_ranking(1)
        #print(feature_ranking_dict)

        features_tmp= list()
        for i in range(len(feature_ranking_dict)):
            features_tmp.append(feature_ranking_dict[i][0]) 
        features = list()
        for x in range(1,30,3):
            features.append(features_tmp[:x])


        p.plot_classifiers_accuracy(features)
        #print(features)
    if run == 'top k features':
        NB = randomForest(df)
        classifier = NB
        classifier.splitTrainTest(train_size = train_size)
        classifier.estimatorGeneration()
        classifier.plot_feature_ranking(feature_to_select = 1,plot = False)
        classifier.predict()

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
    