import numpy as np
import pandas as pd


dataset = pd.read_csv('train.csv',index_col=0)
cols = dataset.columns.tolist()
cols = cols[0:5] + cols[9:11] + cols[5:9] + cols[11:]
dataset = dataset[cols]
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer

imputer_frequent = SimpleImputer(missing_values=np.nan,strategy = 'most_frequent')
X[:,0:4] = imputer_frequent.fit_transform(X[:,0:4])

imputer_no = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value='No')
X[:,4] = imputer_no.fit_transform(X[:,4].reshape(-1,1)).reshape(-1)


imputer_0 = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value=0.0)
X[:,5] = imputer_0.fit_transform(X[:,5].reshape(-1,1)).reshape(-1)

imputer_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
X[:,9:11] = imputer_mean.fit_transform(X[:,9:11])



from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

X[:,0:7] = encoder.fit_transform(X[:,0:7])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:,7:11] = sc.fit_transform(X[:,7:11])

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy')
classifier.fit(X,Y)

####Test set

testset = pd.read_csv('test.csv',index_col=0)
testset = testset[cols[:11]]
X_test = testset.values

X_test[:,0:4] = imputer_frequent.transform(X_test[:,0:4])
X_test[:,4] = imputer_no.transform(X_test[:,4].reshape(-1,1)).reshape(-1)
X_test[:,5] = imputer_0.transform(X_test[:,5].reshape(-1,1)).reshape(-1)
X_test[:,9:11] = imputer_mean.transform(X_test[:,9:11])

X_test[:,0:7] = encoder.transform(X_test[:,0:7])
X_test[:,7:11] = sc.transform(X_test[:,7:11])

Y_pred = classifier.predict(X_test)


final = pd.read_csv('test.csv',usecols=["Loan_ID"])
final['Loan_Status'] = Y_pred

final.to_csv('Random_forest.csv',index=False)