# imports
from warnings import simplefilter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.svm import SVC



# ignoring future warning for tuples/ defining svc
simplefilter(action='ignore', category=FutureWarning)
svc = SVC()

# loading data/checking shape and printing head
df = pd.read_csv('C:/Users/mosaa/Downloads/creditcard.csv')
print(df.shape)
print(df.head())

#checking columns names
col_names = df.columns
print(col_names)
# check distribution/ view the percentage
print(df['Class'].value_counts())
print(df['Class'].value_counts() / float(len(df)))

#summary and checking for missing values
print(df.info())
print(df.isnull().sum())
print(round(df.describe(), 2))

#declaring features, and splitting data
X = df.drop(['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#checking shape and scaling
print(X_train.shape, X_test.shape)
cols = X_train.columns
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.describe())

# fit data
history=svc.fit(X_train, y_train)

# make predictions on test set/ compute and print accuracy score
y_pred = svc.predict(X_test)

print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

# confusion matrix/precision
cm = confusion_matrix(y_test, y_pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


#ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting if a transation was fradulent')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')


plt.show()