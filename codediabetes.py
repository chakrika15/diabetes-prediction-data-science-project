import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import classification_report


#DATA EXPLORATION 
os.chdir("D:\Diabetes")
diabetes_data=pd.read_csv('diabetes.csv')
print("Diabetes data set dimensions: ",diabetes_data.shape)
print("Diabetes data set columns: ",diabetes_data.columns)
print("Diabetes data set data types of variables:\n",diabetes_data.dtypes)
summary_num=diabetes_data.describe() #summary of numerical values
print("Diabetes data set summary: \n",summary_num)
diabetes_data.head() #to explore the data
print("Outcomes from the data set (0/1)\n",diabetes_data['Outcome'].value_counts())
 
#DATA CLEANING
data=diabetes_data.copy()
data.columns
data.isnull().sum() #count of null values 
np.unique(data['Pregnancies']) 
np.unique(data['Glucose']) # a person cannot have glucose level of 0
np.unique(data['BloodPressure']) # a person cannot have BP level of 0
np.unique(data['SkinThickness']) # a person cannot have SkinThickness level of 0
np.unique(data['Insulin']) # a person cannot have insulin level of 0
np.unique(data['BMI']) # a person cannot have BMI of 0
np.unique(data['DiabetesPedigreeFunction']) 
np.unique(data['Age']) 
print('Total no. of entries having 0 glucose level: ',data[data.Glucose==0].shape[0])
print('Total no. of entries having 0 BP level     : ',data[data.BloodPressure==0].shape[0])
print('Total no. of entries having 0 SkinThickness: ',data[data.SkinThickness==0].shape[0])
print('Total no. of entries having 0 Insulin level: ',data[data.Insulin==0].shape[0])
print('Total no. of entries having 0 BMI level    : ',data[data.BMI==0].shape[0])
data1 = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose != 0)]
data1.shape

#DATA VISUALIZATIONS 
# I Frequency distributions: 
sns.distplot(data1['Pregnancies'],kde=False,bins=15)
sns.distplot(data1['Glucose'],kde=False,bins=15)
sns.distplot(data1['BloodPressure'],kde=False,bins=15)
sns.distplot(data1['SkinThickness'],kde=False,bins=15)
sns.distplot(data1['Insulin'],kde=False,bins=15)
sns.distplot(data1['BMI'],kde=False,bins=15)
sns.distplot(data1['DiabetesPedigreeFunction'],kde=False,bins=15)
sns.distplot(data1['Age'],kde=False,bins=15)
sns.boxplot(data1['Age'])
sns.countplot(data1['Outcome'])

# II Correlation:
correlation=data1.corr()

# III relationship btw the variables vs the outcomes 
data1['Pregnancies'].hist(by=data1['Outcome'],bins=10)
data1['Glucose'].hist(by=data1['Outcome'],bins=10)
data1['BloodPressure'].hist(by=data1['Outcome'],bins=10)
data1['SkinThickness'].hist(by=data1['Outcome'],bins=10)
data1['Insulin'].hist(by=data1['Outcome'],bins=10)
data1['BMI'].hist(by=data1['Outcome'],bins=10)
data1['DiabetesPedigreeFunction'].hist(by=data1['Outcome'],bins=10)
data1['Age'].hist(by=data1['Outcome'],bins=10)


#FEATURE SELECTION
columns_list=list(data1.columns)
print(columns_list)
features=list(set(columns_list)-set(['Outcome']))
print(features)
x=data1[features].values
print(x)
y=data1['Outcome'].values
print(y)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#LOGISTIC REGRESSION
logistic=LogisticRegression()
logistic.fit(train_x,train_y)
prediction_lr=logistic.predict(test_x)
print(prediction_lr)
acc_lr=accuracy_score(test_y,prediction_lr)
print(acc_lr)
ps_lr=precision_score(test_y,prediction_lr)
print(ps_lr)
confusion_matrix_lr=confusion_matrix(test_y,prediction_lr)
print(confusion_matrix_lr)
print(classification_report(test_y, prediction_lr))


#KNN
knn = KNeighborsClassifier()
knn.fit(train_x, train_y)
prediction_knn=knn.predict(test_x)
print(prediction_knn)
acc_knn=accuracy_score(test_y,prediction_knn)
print(acc_knn)
ps_knn=precision_score(test_y,prediction_knn)
print(ps_knn)
print(classification_report(test_y, prediction_knn))
confusion_matrix_knn=confusion_matrix(test_y,prediction_knn)
print(confusion_matrix_knn)

#RANDOM FOREST CLASSIFIER 
rfc=RandomForestClassifier()
rfc.fit(train_x,train_y)
prediction_rfc=rfc.predict(test_x)
print(prediction_rfc)
acc_rfc=accuracy_score(test_y,prediction_rfc)
print(acc_rfc)
ps_rfc=precision_score(test_y,prediction_rfc)
print(ps_rfc)
print(classification_report(test_y, prediction_rfc))
confusion_matrix_rfc=confusion_matrix(test_y,prediction_rfc)
print(confusion_matrix_rfc)

#Saving RFC Model
Pkl_Filename = "Pickle_RFC_Model.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rfc, file)
    
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_RFC1_Model = pickle.load(file)

Pickled_RFC1_Model

score = Pickled_RFC1_Model.score(test_x, test_y)  
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_RFC1_Model.predict(test_x)  
Ypredict







