import pandas as pd
#reading train and test csvs
train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')
#dropping survived column from x_train
X_train = train_df.drop("Survived",axis=1)
#setting Y_train equal to train_df data
Y_train = train_df["Survived"]
#setting X_test to test_df, dropping PassengerID
X_test = test_df.drop("PassengerId",axis=1)
#Printing the mean of the survival rate grouped by sex, 1 = woman, 0 = man
print(train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False))

#Since the survival rate for women is more than 4x higher than the men's it is important that we do keep this feature as it is significant.
