import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
data= pd.read_csv("C:\\Users\\ravi0\\OneDrive\\Documents\\cleaned_ingredients2.csv")
data= data.drop(data[['NDB_No','Descrip']], axis=1)
 
data['Magnesium_mg']=pd.to_numeric(data['Magnesium_mg'], errors='coerce')
data.dropna(subset='Magnesium_mg')
data
x= data.iloc[ : , : 7]
print(x)
y= data['Diabetic_Friendly']

sr= StandardScaler()
sr.fit(x)
x=sr.transform(x)
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100, max_depth=5,  random_state=42)
model.fit(x_train,y_train)

x_train_pred= model.predict(x_train)
x_accuracy= accuracy_score(y_train,x_train_pred)
x_accuracy
x_test_pred= model.predict(x_test)
xt_accuracy= accuracy_score(y_test,x_test_pred)
xt_accuracy

def predict_diabetes (input_data):
    input_data= np.asarray(input_data)
    input_data=input_data.reshape(1,-1)
    input_data=sr.transform(input_data)
    prediction= model.predict(input_data)
    return prediction[0]

