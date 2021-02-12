import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
#reading from file and setting train and test
df = pd.read_csv('glass.csv')
#dropping type from train
train_df = df.drop('Type', axis=1)
test_df = df['Type']
#splitting the test and train elements
X_train,X_test,Y_train,Y_test = train_test_split(train_df,test_df, test_size=0.3)
#printing all the train features
print(train_df[train_df.isnull().any(axis=1)])

#SVM
#Setting svm to svc()
svc = SVC()
#fitting X_train and Y_train data to svm
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
#calculating svc
acc_svc = round(svc.score(X_train, Y_train)*100,2)
#output
print ("SVM score is: ", acc_svc)


#bayes test was more accurate since there might not be a firm line between data sets
