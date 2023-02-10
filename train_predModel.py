from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('final_result.csv')

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"(X_test.shape[0], (y_test != y_pred).sum()))