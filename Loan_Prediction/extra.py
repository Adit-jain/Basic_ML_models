import pandas as pd


sample = pd.read_csv("sample.csv")
Rand = pd.read_csv("Random_forest.csv",usecols=["Loan_ID","Loan_Status"])

Rand.to_csv("Random.csv")

Rand_2 = pd.read_csv("Random.csv")