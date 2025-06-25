import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix

dataframe = pd.read_csv('/Users/mariasolovej/Documents/prog/sberkonk/train.csv') 
test = pd.read_csv("/Users/mariasolovej/Documents/prog/sberkonk/test.csv")

def convert(dataframe):
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"],errors="coerce")
    dataframe = dataframe.drop(columns=["id"],errors='ignore')
    columns = ["Churn","Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"]
    dataframe[columns] = dataframe[columns].replace({"Yes":1,"No":0,"No internet service":2,"No phone service":2})
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors='coerce').fillna(0)
    dataframe = pd.get_dummies(dataframe).astype(int)
    return dataframe

dataframe = convert(dataframe)
test = convert(dataframe)
dataframe.to_csv("new_file.csv",index=False)
dataframe.to_html("preview.html")

signs = dataframe.drop("Churn", axis=1)
goal = dataframe["Churn"]


SignsTrain, SignsTest, GoalTrain, GoalTest = train_test_split(signs,goal,test_size=0.2,random_state=42,stratify=goal)

model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(SignsTrain,GoalTrain)

PredGoal = model.predict(SignsTest)

print("Classification report:","\n", classification_report(GoalTest,PredGoal))
print("confusion matrix:","\n",confusion_matrix(GoalTest,PredGoal))