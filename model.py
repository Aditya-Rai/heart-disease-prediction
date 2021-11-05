# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:55:34 2021

@author: Aditya
"""

#importing Libraries

# Importing the important libraries

# Importing the important libraries

import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.preprocessing import PowerTransformer

import pickle
import pylab
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset

df = pd.read_csv("heart.csv")
df.head()

# Seperating the dependent and the independent columns

X = df.drop("HeartDisease" , axis = 1)
y = df['HeartDisease']


# Seperating the numerical and the categorical columns

num_cols = [col for col in X.columns if X[col].dtype != "object"]
cat_cols = [col for col in X.columns if col not in num_cols]

# Handling the outlier of the Normal columns
normal_cols = ["Age","MaxHR","RestingBP"]
for col in normal_cols:
    mean = np.mean(df[col])
    std = np.std(df[col])
    lower_range = mean - (3*std)
    upper_range = mean + (3*std)
    df[col] = np.where(((df[col] < lower_range) | (df[col] > upper_range))
                            ,random.randint(int(lower_range),int(upper_range)),df[col])
    

# Handling the ouliers of columns not following normal distribution
IQR = np.percentile(df["Cholesterol"],75) - np.percentile(df["Cholesterol"],25)
lower_bound = np.percentile(df["Cholesterol"],25) - 1.5 * IQR
upper_bound = np.percentile(df["Cholesterol"],75) + 1.5 * IQR
median_cholesterol = np.median(df["Cholesterol"])

df["Cholesterol"] = np.where(((df["Cholesterol"] > upper_bound) | (df["Cholesterol"] < lower_bound)) 
                                 ,random.randint(int(np.percentile(df["Cholesterol"],25)),
                                                 int(np.percentile(df["Cholesterol"],75))),df["Cholesterol"])

    

# Seperating the train and test dataset 

x_train,x_test,y_train,y_test = train_test_split(X,y,stratify = y , random_state=42,test_size=0.2)

scaler = StandardScaler()
scaler.fit(x_train[num_cols])
x_train[num_cols] = scaler.transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])


x_train = pd.get_dummies(data = x_train , drop_first=True)
x_test = pd.get_dummies(data = x_test , drop_first = True)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)

# Saving model to disk
pickle.dump(knn, open('model.pkl','wb'))


























