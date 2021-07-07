import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d1 = pd.read_csv("winequality-white.csv",sep='[:;]',engine='python')
d1.columns = d1.columns.str.replace('"','')
d1['Class'] = 'White'
d2 = pd.read_csv("winequality-red.csv",sep='[:;]',engine='python')
d2.columns = d2.columns.str.replace('"','')
d2['Class'] = 'Red'

dataset = pd.concat([d1,d2],ignore_index=True)

from sklearn.utils import shuffle

dataset = shuffle(dataset)

dataset.reset_index(inplace=True,drop=True)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)

from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train_scaled,Y_train)

Y_pred = classifier.predict(sc.transform(X_test))

print(np.concatenate((Y_test.reshape(len(Y_test),1),Y_pred.reshape(len(Y_pred),1)),1))


from sklearn.metrics import confusion_matrix,accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))

