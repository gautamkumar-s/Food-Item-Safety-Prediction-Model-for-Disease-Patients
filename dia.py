import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def predict_diabetes (input_data):
    data= pd.read_csv("my_dataset1.csv")
    data= data.drop(data[['NDB_No','Descrip']], axis=1)

    
    data['Magnesium_mg']=pd.to_numeric(data['Magnesium_mg'], errors='coerce')
    data.dropna(subset='Magnesium_mg')
    data

    x= data.iloc[ : , : 6]

    y= data['Diabetic_Friendly']

    sr= StandardScaler()
    sr.fit(x)
    x1=sr.transform(x)

    x_train, x_test, y_train, y_test= train_test_split(x1,y, test_size=0.2,random_state=42)
    model=RandomForestClassifier(n_estimators=100, max_depth=5,  random_state=42)
    model.fit(x_train,y_train)

    x_train_pred= model.predict(x_train)
    x_accuracy= accuracy_score(y_train,x_train_pred)
    print(x_accuracy)
    x_test_pred= model.predict(x_test)
    xt_accuracy= accuracy_score(y_test,x_test_pred)
    print(xt_accuracy)
    input_data= np.asarray(input_data)
    input_data=input_data.reshape(1,-1)
    input_data = sr.transform(input_data)  
    prediction=model.predict(input_data)
    return prediction[0]

def predict_bp (input_data):
    data= pd.read_csv("my_dataset2.csv")
    x=data.iloc[:,1:7]
    y=data['High_BP_Friendly']
    sr=StandardScaler()
    sr.fit(x)
    x=sr.transform(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    rand=RandomForestClassifier(max_depth=5,n_estimators=100,random_state=42)
    rand.fit(x_train,y_train)
    pred_train= rand.predict(x_train)
    training_accuracy= accuracy_score(pred_train,y_train)
    #print(training_accuracy)
    pred_test= rand.predict(x_test)
    testing_accuracy= accuracy_score(pred_test,y_test)
    testing_accuracy

    input_data= np.asarray(input_data)
    input_data=input_data.reshape(1,-1)
    input_data = sr.transform(input_data) 
    prediction=rand.predict(input_data)
    return prediction
    
def predict_obesity(input_data):
    from sklearn.model_selection import cross_val_score
    
    data= pd.read_csv("my_dataset1.csv")
    data= data.drop(data[['NDB_No','Descrip']], axis=1)
    data['Magnesium_mg']=pd.to_numeric(data['Magnesium_mg'], errors='coerce')
    data.dropna(subset='Magnesium_mg')
    x=data[['Energy_kcal','Saturated_fats_g','Fat_g','Carb_g','Sugar_g','Calcium_mg','Iron_mg','Magnesium_mg','Sodium_mg']]
    y=data['obesity']
    y.value_counts()
    sr=StandardScaler()
    sr.fit(x)
    x=sr.transform(x)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    rnd=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
    rnd.fit(x_train,y_train)
    training_pred=rnd.predict(x_train)
    train_accuracy=accuracy_score(y_train,training_pred)
    train_accuracy 
    testing_pred=rnd.predict(x_test)
    test_accuracy=accuracy_score(y_test,testing_pred)
    test_accuracy 
    input_data= np.asarray(input_data)
    input_data=input_data.reshape(1,-1)
    input_data = sr.transform(input_data) 
    prediction=rnd.predict(input_data)
    return prediction
    

gg=(116	,0.1	,0.4	,20.0	,0.3	,19	,3.3	,36	,2)

aa=predict_obesity(gg)
print(aa)