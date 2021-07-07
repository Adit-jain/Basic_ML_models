import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

from sklearn.model_selection import cross_val_score

score= cross_val_score(classifier,X_train,Y_train,cv=10)
print("Accuracy : {:.2f} %".format(score.mean()*100))
print("Standard Deviation : {:.2f} %".format(score.std()*100))


plt.scatter(range(1,21),Y_test[:20],color='red')
plt.plot(range(1,21),Y_pred[:20],color='blue')
plt.title("Logisti regression on Breast cancer")
plt.xlabel("Data")
plt.ylabel("Class")
plt.show()
