import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

dataframe = pd.read_csv('binary_classification/sklearn/train.csv') 
test = pd.read_csv("binary_classification/sklearn/test.csv")

def convert(dataframe):
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"],errors="coerce")
    dataframe = dataframe.drop(columns=["id"],errors='ignore')
    
    columns = ["Churn","Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"]
    dataframe[columns] = dataframe[columns].replace({"Yes":1,"No":0,"No internet service":2,"No phone service":2})
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors='coerce').fillna(0)
    dataframe["AvgMonthlyCharges"] = dataframe["TotalCharges"]/(dataframe["tenure"]+1)
    dataframe["IsSenior"] = dataframe["SeniorCitizen"]*(1-dataframe["Partner"])
    # dataframe["AvgChargePerMonth"] = dataframe["TotalCharges"]/(dataframe["tenure"]+1)
    # dataframe["NewHighPayer"] = ((dataframe["tenure"] < 6) & (dataframe["MonthlyCharges"] > dataframe["MonthlyCharges"].median())).astype(int)
    dataframe["NumServices"] = dataframe[[
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]].eq(1).sum(axis=1)
    dataframe["HasStreaming"] = ((dataframe["StreamingTV"] == 1) | (dataframe["StreamingMovies"] == 1)).astype(int)
   

    


    



    
    
    dataframe["HasPhoneLines"] = (dataframe["PhoneService"] != 0).astype(int)
    dataframe = pd.get_dummies(dataframe).astype(int)
    dataframe["IsFiberUser"] = dataframe.get("InternetService_Fiber optic", 0)
    dataframe["HasInternetButNoServices"] = ((dataframe["NumServices"] == 0) & (dataframe["InternetService_Fiber optic"] == 1)).astype(int)
    return dataframe

dataframe = convert(dataframe)
test = convert(dataframe)
dataframe.to_csv("new_file.csv",index=False)
dataframe.to_html("preview.html")

signs = dataframe.drop("Churn", axis=1)
goal = dataframe["Churn"]


SignsTrain, SignsTest, GoalTrain, GoalTest = train_test_split(signs,goal,test_size=0.2,random_state=42,stratify=goal)

# model = RandomForestClassifier(
#     n_estimators=3234, #234 - 777
#     max_depth=15,
#     min_samples_split=5,
#     random_state=20,
#     min_samples_leaf=6,
#     class_weight="balanced")

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=20
)


hgb = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=20
)

lr = LogisticRegression(max_iter=1000,C=1.8)

ensemble = VotingClassifier(
    estimators=[("lr", lr),("gb",gb),("hgb",hgb)],
    voting='soft',
    weights=[3,9,2]
)

# model.fit(SignsTrain,GoalTrain)
ensemble.fit(SignsTrain,GoalTrain)
PredGoal = ensemble.predict(SignsTest)


print("Classification report:","\n", classification_report(GoalTest,PredGoal))
print("confusion matrix:","\n",confusion_matrix(GoalTest,PredGoal))

f1 = f1_score(GoalTest,PredGoal,average="weighted")
print("weighted F1-score:", round(f1,4))