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
print("\033c")

dataframe = pd.read_csv('binary_classification/sklearn/train.csv') 
test = pd.read_csv("binary_classification/sklearn/test.csv")
test_ids = []

def convert(dataframe):
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"],errors="coerce")
    
    
    columns = ["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"]
    if "Churn" in dataframe.columns:
        columns.append("Churn")
        
    dataframe[columns] = dataframe[columns].replace({"Yes":1,"No":0,"No internet service":2,"No phone service":2})
    dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors='coerce').fillna(0)
    dataframe["AvgMonthlyCharges"] = dataframe["TotalCharges"]/(dataframe["tenure"]+1)
    dataframe["IsSenior"] = dataframe["SeniorCitizen"]*(1-dataframe["Partner"])
    dataframe["NumServices"] = dataframe[[
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]].eq(1).sum(axis=1)
    dataframe["HasStreaming"] = ((dataframe["StreamingTV"] == 1) | (dataframe["StreamingMovies"] == 1)).astype(int)
   

   
    dataframe["HasPhoneLines"] = (dataframe["PhoneService"] != 0).astype(int)
    dataframe = pd.get_dummies(dataframe).astype(int)
    dataframe["IsFiberUser"] = dataframe.get("InternetService_Fiber optic", 0)
    dataframe["HasInternetButNoServices"] = ((dataframe["NumServices"] == 0) & (dataframe["InternetService_Fiber optic"] == 1)).astype(int)
    test_ids = dataframe["id"]
    dataframe = dataframe.drop(columns=["id"],errors='ignore')
    return dataframe,test_ids

dataframe,dfid = convert(dataframe)
test,test_ids = convert(test)
dataframe.to_csv("new_file.csv",index=False)
dataframe.to_html("preview.html")

signs = dataframe.drop("Churn", axis=1)
goal = dataframe["Churn"]


SignsTrain, SignsTest, GoalTrain, GoalTest = train_test_split(signs,goal,test_size=0.2,random_state=42,stratify=goal)


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

ensemble.fit(SignsTrain,GoalTrain)
PredGoal = ensemble.predict(SignsTest)

prediction = ensemble.predict(test)
results = pd.DataFrame({
    "id": test_ids,
    "Churn": prediction
}
)
results.sort_values("id")
results.to_html("binary_classification/sklearn/results.html",index=False)

# f1 = f1_score(GoalTest,PredGoal,average="weighted")
# print("weighted F1-score:", round(f1,4))