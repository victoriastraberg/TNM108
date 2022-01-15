
# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns
import matplotlib.pyplot as plt
 
# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv" 
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv" 
test = pd.read_csv(test_url)

print("***** Train_Set *****") 
print(train.head())
print("\n")
print("***** Test_Set *****") 
print(test.head())

print("***** Train_Set describe *****") 
print(train.describe())

print("***** Test_Set  *****") 
print(test.describe())

print(train.columns.values)

# For the train set 
print("***** TRAIN_ISNA  *****") 
print(train.isna().head()) # isna = is used to detect missing values 
print("\n")
# For the test set 
print(test.isna().head())

print("*****In the train set*****") 
print(train.isna().sum())
print("\n")
print("*****In the test set*****") 
print(test.isna().sum())

# Fill missing values with mean column values in the train set
print("FILLNA")
train.fillna(train.mean(), inplace=True)
print("\n")
# Fill missing values with mean column values in the test set 
test.fillna(test.mean(), inplace=True)

print(train.isna().sum())

print(test.isna().sum())

train['Ticket'].head()

train['Cabin'].head()

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train, col='Survived') 
g.map(plt.hist, 'Age', bins=20) 
#<seaborn.axisgrid.FacetGrid at 0x7fa990f87080>

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6) 
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# Show graphs 
# plt.show() 

# Some are numerical and some are not
train.info()
#<class 'pandas.core.frame.DataFrame'>

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Label Encoding 
print("Label Encoding")
labelEncoder = LabelEncoder() 
labelEncoder.fit(train['Sex']) 
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex']) 
test['Sex'] = labelEncoder.transform(test['Sex'])

# Let's investigate if you have non-numeric data left
train.info()
#<class 'pandas.core.frame.DataFrame'>

test.info()
#<class 'pandas.core.frame.DataFrame'>

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])
print("Hejsan") # DROP FUNCTION SURVIVAL 
print(train.info())

#<class 'pandas.core.frame.DataFrame'>

# Let's build the K-Means model
kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me) 
    if prediction[0] == y[i]:
     correct += 1

print(correct/len(X)) #0.5084175084175084


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0) 

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me)) 
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))

