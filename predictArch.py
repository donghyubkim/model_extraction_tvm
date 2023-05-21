
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
        

    def classifier_train(self):
        self.classifiers = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
        for clf in self.classifiers:
            clf.fit(self.X_train,self.Y_train)

    def classifier_predict(self,df):
        target_Y = df['label_model_name'].to_numpy()
        target_X = df.drop(['label_model_name'], axis = 'columns').to_numpy()
        
        acc_list = []
        for clf in self.classifiers:
            self.predicted=clf.predict(target_X)
            self.acc = accuracy_score(target_Y, self.predicted)
            acc_list.append((type(clf).__name__,self.acc))
            
        return acc_list
    
    def classifier_predict_binning(self,X,Y):
        
        
        acc_list = []
        for clf in self.classifiers:
            self.predicted=clf.predict(X)
            self.acc = accuracy_score(Y, self.predicted)
            acc_list.append((type(clf).__name__,self.acc))
            
        return acc_list

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

    def binning(self,normalization = True):
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
        if normalization:
            self.X = self.normalization.fit_transform(self.X)
        else:
            pass
        
            
        print(self.X)

        return self.X , self.Y


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
        plt.savefig("learning_curve_multiopt_binning", dpi=500, bbox_inches="tight")
        plt.show()

   
    def plot_classifiers_accuracy(self,column_sets):

        
        classifiers = [GaussianNB(),LogisticRegression(),RandomForestClassifier(),MLPClassifier(),NearestCentroid(),KNeighborsClassifier(),AdaBoostClassifier()]
        plt.figure(figsize=(5,4))
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
        plt.xticks([1,15,30,45,60,75,90,105,120,135,146])
        plt.ylabel('Architecture Prediction Accuracy')
        plt.title('')
        plt.legend(loc = "lower right")
        plt.tight_layout()
        plt.savefig("new_all_opt_scenario1_optimized_faeture_num", dpi=500, bbox_inches="tight")
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
    
    run = "multi_opt_level_learning_curve"
    df = pd.read_csv('./golden/optlevel0/pred_model_trainable_data.csv')
    
    train_size = 0.5

    if run == "multi_opt_level_learning_curve":

        df0 =  pd.read_csv('./golden/optlevel0/pred_model_trainable_data.csv')
        df0["label_model_name"] = df0["label_model_name"] + "_opt0"
        df1 =  pd.read_csv('./golden/optlevel1/pred_model_trainable_data.csv')
        df1["label_model_name"] = df1["label_model_name"] + "_opt1"
        df2 =  pd.read_csv('./golden/optlevel2/pred_model_trainable_data.csv')
        df2["label_model_name"] = df2["label_model_name"] + "_opt2"
        df3 =  pd.read_csv('./golden/optlevel3/pred_model_trainable_data.csv')
        df3["label_model_name"] = df3["label_model_name"] + "_opt3"
        # merge df
        merged_df = pd.merge(df0, df1, how= "outer")
        merged_df = pd.merge(merged_df, df2, how= "outer")
        merged_df = pd.merge(merged_df, df3, how= "outer")
        merged_df.fillna(0,inplace = True)

        #to fit to the classifier to match the missing columns
        new_df0 = merged_df[merged_df['label_model_name'].str.endswith('opt0')]
        new_df1 = merged_df[merged_df['label_model_name'].str.endswith('opt1')]
        new_df2 = merged_df[merged_df['label_model_name'].str.endswith('opt2')]
        new_df3 = merged_df[merged_df['label_model_name'].str.endswith('opt3')]

        

        p = predictBase(merged_df,normalization=False)
        p.splitTrainTest(train_size)
        p.classifier_train()
        #p.plot_learning_curve()
        '''
        acc_opt0 = p.classifier_predict(new_df0)
        acc_opt1 =p.classifier_predict(new_df1)
        acc_opt2 =p.classifier_predict(new_df2)
        acc_opt3 =p.classifier_predict(new_df3)

        print("opt0",acc_opt0)
        print("opt1",acc_opt1)
        print("opt2",acc_opt2)
        print("opt3",acc_opt3)
        '''
        _,_,feature_ranking_dict = p.plot_feature_ranking(1)
        #print(feature_ranking_dict)

        features_tmp= list()
        for i in range(len(feature_ranking_dict)):
            features_tmp.append(feature_ranking_dict[i][0]) 
        features = list()
        for x in range(1,len(features_tmp),2):
            features.append(features_tmp[:x])
        p.plot_classifiers_accuracy(features)
    if run == "across_opt_level_experiment":
        df0 =  pd.read_csv('./golden/optlevel0/pred_model_trainable_data.csv')
        df1 =  pd.read_csv('./golden/optlevel1/pred_model_trainable_data.csv')
        df2 =  pd.read_csv('./golden/optlevel2/pred_model_trainable_data.csv')
        df3 =  pd.read_csv('./golden/optlevel3/pred_model_trainable_data.csv')

        p0 = predictBase(df0,normalization=False)
        binned_X_opt0,binned_Y_opt0  = p0.binning(normalization=False)
        p0.splitTrainTest(train_size)
        p0.classifier_train()

        p1 = predictBase(df1,normalization=False)
        binned_X_opt1,binned_Y_opt1  = p1.binning(normalization=False)
        p1.splitTrainTest(train_size)
        p1.classifier_train()

        p2 = predictBase(df2,normalization=False)
        binned_X_opt2,binned_Y_opt2 = p2.binning(normalization=False)
        p2.splitTrainTest(train_size)
        p2.classifier_train()

        p3 = predictBase(df3,normalization=False)
        binned_X_opt3,binned_Y_opt3  = p3.binning(normalization=False)
        p3.splitTrainTest(train_size)
        p3.classifier_train()

        merged_df03 = pd.merge(df0, df3, how= "outer")
        merged_df03.fillna(0,inplace = True)

        merged_df02 = pd.merge(df0, df2, how= "outer")
        merged_df02.fillna(0,inplace = True)

        merged_df13 = pd.merge(df1, df3, how= "outer")
        merged_df13.fillna(0,inplace = True)

        merged_df01 = pd.merge(df0, df1, how= "outer")
        merged_df01.fillna(0,inplace = True)

        merged_df12 = pd.merge(df1, df2, how= "outer")
        merged_df12.fillna(0,inplace = True)

        merged_df23 = pd.merge(df2, df3, how= "outer")
        merged_df23.fillna(0,inplace = True)

        p03 = predictBase(merged_df03,normalization=False)
        binned_X_opt03,binned_Y_opt03  = p03.binning(normalization=False)
        p03.splitTrainTest(train_size)
        p03.classifier_train()

        p02 = predictBase(merged_df02,normalization=False)
        binned_X_opt02,binned_Y_opt02  = p02.binning(normalization=False)
        p02.splitTrainTest(train_size)
        p02.classifier_train()

        p01 = predictBase(merged_df01,normalization=False)
        binned_X_opt01,binned_Y_opt01  = p01.binning(normalization=False)
        p01.splitTrainTest(train_size)
        p01.classifier_train()

        p12 = predictBase(merged_df12,normalization=False)
        binned_X_opt12,binned_Y_opt12  = p12.binning(normalization=False)
        p12.splitTrainTest(train_size)
        p12.classifier_train()

        p23 = predictBase(merged_df23,normalization=False)
        binned_X_opt23,binned_Y_opt23  = p23.binning(normalization=False)
        p23.splitTrainTest(train_size)
        p23.classifier_train()

        p13 = predictBase(merged_df13,normalization=False)
        binned_X_opt13,binned_Y_opt13  = p13.binning(normalization=False)
        p13.splitTrainTest(train_size)
        p13.classifier_train()
        
        acc_opt031 =p03.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt032 =p03.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        
        acc_opt021 =p02.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt023 =p02.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt130 =p13.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt132 =p13.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)


        acc_opt00 =p0.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt01 =p0.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt02 =p0.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        acc_opt03 =p0.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt10 =p1.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt12 =p1.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        acc_opt13 =p1.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt20 =p2.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt21 =p2.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt23 =p2.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt30 =p3.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt31 =p3.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt32 =p3.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)

        acc_opt012 =p01.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        acc_opt013 =p01.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt120 =p12.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt123 =p12.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt230 =p23.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt231 =p23.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        
        acc_opt030 =p03.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt033 =p03.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)
        
        acc_opt020 =p02.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt022 =p02.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)

        acc_opt131 =p13.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt133 =p13.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt010 =p01.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt011 =p01.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)

        acc_opt121 =p12.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt122 =p12.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)

        acc_opt232 =p23.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        acc_opt233 =p23.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        acc_opt11 =p1.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt22 =p2.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        acc_opt33 =p3.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        print("opt00",acc_opt00)
        print("opt01",acc_opt01)
        print("opt02",acc_opt02)
        print("opt03",acc_opt03)

        print("opt10",acc_opt10)
        print("opt12",acc_opt12)
        print("opt13",acc_opt13)

        print("opt20",acc_opt20)
        print("opt21",acc_opt21)
        print("opt23",acc_opt23)

        print("opt30",acc_opt30)
        print("opt31",acc_opt31)
        print("opt32",acc_opt32)

        print("opt031",acc_opt031)
        print("opt032",acc_opt032)

        print("opt021",acc_opt021)
        print("opt023",acc_opt023)

        print("opt130",acc_opt130)
        print("opt132",acc_opt132)
        
        print("opt012",acc_opt012)
        print("opt013",acc_opt013)
        print("opt120",acc_opt120)
        print("opt123",acc_opt123)
        print("opt230",acc_opt230)
        print("opt231",acc_opt231)

        print("opt030",acc_opt030)
        print("opt033",acc_opt033)
        print("opt020",acc_opt020)
        print("opt022",acc_opt022)
        print("opt131",acc_opt131)
        print("opt133",acc_opt133)

        print("opt010",acc_opt010)
        print("opt011",acc_opt011)
        print("opt121",acc_opt121)
        print("opt122",acc_opt122)
        print("opt232",acc_opt232)
        print("opt233",acc_opt232)

        print("opt00",acc_opt00)
        print("opt11",acc_opt11)
        print("opt22",acc_opt22)
        print("opt33",acc_opt33)

    if run == "multi_opt_level_test_generic":

        df0 =  pd.read_csv('./golden/optlevel0/pred_model_trainable_data.csv')
        df0["label_model_name"] = df0["label_model_name"] + "_opt0"
        df1 =  pd.read_csv('./golden/optlevel1/pred_model_trainable_data.csv')
        df1["label_model_name"] = df1["label_model_name"] + "_opt1"
        df2 =  pd.read_csv('./golden/optlevel2/pred_model_trainable_data.csv')
        df2["label_model_name"] = df2["label_model_name"] + "_opt2"
        df3 =  pd.read_csv('./golden/optlevel3/pred_model_trainable_data.csv')
        df3["label_model_name"] = df3["label_model_name"] + "_opt3"
        # merge df
        merged_df = pd.merge(df0, df1, how= "outer")
        merged_df = pd.merge(merged_df, df2, how= "outer")
        merged_df = pd.merge(merged_df, df3, how= "outer")
        merged_df.fillna(0,inplace = True)

        #to fit to the classifier to match the missing columns
        new_df0 = merged_df[merged_df['label_model_name'].str.endswith('opt0')]
        new_df1 = merged_df[merged_df['label_model_name'].str.endswith('opt1')]
        new_df2 = merged_df[merged_df['label_model_name'].str.endswith('opt2')]
        new_df3 = merged_df[merged_df['label_model_name'].str.endswith('opt3')]

        

        p = predictBase(merged_df,normalization=False)
        p.splitTrainTest(train_size)
        p.classifier_train()

        acc_opt0 = p.classifier_predict(new_df0)
        acc_opt1 =p.classifier_predict(new_df1)
        acc_opt2 =p.classifier_predict(new_df2)
        acc_opt3 =p.classifier_predict(new_df3)

        print("opt0",acc_opt0)
        print("opt1",acc_opt1)
        print("opt2",acc_opt2)
        print("opt3",acc_opt3)
    
    if run == "multi_opt_level_test_binning":
        df0 =  pd.read_csv('./golden/optlevel0/pred_model_trainable_data.csv')
        df0["label_model_name"] = df0["label_model_name"] + "_opt0"
        df1 =  pd.read_csv('./golden/optlevel1/pred_model_trainable_data.csv')
        df1["label_model_name"] = df1["label_model_name"] + "_opt1"
        df2 =  pd.read_csv('./golden/optlevel2/pred_model_trainable_data.csv')
        df2["label_model_name"] = df2["label_model_name"] + "_opt2"
        df3 =  pd.read_csv('./golden/optlevel3/pred_model_trainable_data.csv')
        df3["label_model_name"] = df3["label_model_name"] + "_opt3"
        # merge df
        merged_df = pd.merge(df0, df1, how= "outer")
        merged_df = pd.merge(merged_df, df2, how= "outer")
        merged_df = pd.merge(merged_df, df3, how= "outer")
        merged_df.fillna(0,inplace = True)

        p0 = predictBase(df0)
        binned_X_opt0,binned_Y_opt0  = p0.binning(normalization=False)
        p1 = predictBase(df1)
        binned_X_opt1,binned_Y_opt1  = p1.binning(normalization=False)
        p2 = predictBase(df2)
        binned_X_opt2,binned_Y_opt2  = p2.binning(normalization=False)
        p3 = predictBase(df3)
        binned_X_opt3,binned_Y_opt3  = p3.binning(normalization=False)
        

        p = predictBase(merged_df,normalization=False)
        p.binning(normalization=False)
        p.splitTrainTest(train_size)
        p.classifier_train()
        p.plot_learning_curve()
        '''
        acc_opt0 = p.classifier_predict_binning(binned_X_opt0,binned_Y_opt0)
        acc_opt1 =p.classifier_predict_binning(binned_X_opt1,binned_Y_opt1)
        acc_opt2 =p.classifier_predict_binning(binned_X_opt2,binned_Y_opt2)
        acc_opt3 =p.classifier_predict_binning(binned_X_opt3,binned_Y_opt3)

        print("opt0",acc_opt0)
        print("opt1",acc_opt1)
        print("opt2",acc_opt2)
        print("opt3",acc_opt3)

        '''




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
        p = predictBase(df.drop(['total_percentage'], axis = 'columns'))
        p.splitTrainTest(train_size= 0.5)
        _,_,feature_ranking_dict = p.plot_feature_ranking(1)
        #print(feature_ranking_dict)

        features_tmp= list()
        for i in range(len(feature_ranking_dict)):
            features_tmp.append(feature_ranking_dict[i][0]) 
        features = list()
        for x in range(1,59,2):
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

    