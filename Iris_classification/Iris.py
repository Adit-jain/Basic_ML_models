import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Iris.csv",index_col=0)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(sc.transform(X_test))

from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))

plt.scatter(X_test[Y_pred=='Iris-virginica',0],X_test[Y_pred=='Iris-virginica',1],color='red',label='Iris-virginica')
plt.scatter(X_test[Y_pred=='Iris-versicolor',0],X_test[Y_pred=='Iris-versicolor',1],color='blue',label = 'Iris-versicolor')
plt.scatter(X_test[Y_pred=='Iris-setosa',0],X_test[Y_pred=='Iris-setosa',1],color='green',label ='Iris-setosa' )
plt.title("Iris Classification")
plt.xlabel("Sepal length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

plt.scatter(X_test[Y_pred=='Iris-virginica',2],X_test[Y_pred=='Iris-virginica',3],color='red',label='Iris-virginica')
plt.scatter(X_test[Y_pred=='Iris-versicolor',2],X_test[Y_pred=='Iris-versicolor',3],color='blue',label = 'Iris-versicolor')
plt.scatter(X_test[Y_pred=='Iris-setosa',2],X_test[Y_pred=='Iris-setosa',3],color='green',label ='Iris-setosa' )
plt.title("Iris Classification")
plt.xlabel("Petal length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()