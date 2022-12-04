import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_predict,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns; sns.set()
import time
from sklearn.feature_selection import VarianceThreshold
from imblearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn.svm import SVC # "Support vector classifier"
import optuna
import sklearn.datasets
from sklearn.datasets import fetch_openml
import sklearn.neural_network

def objective_svm(trial):
    vr = VarianceThreshold(0.0001)
    sc = StandardScaler()

    score = 0
  
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8,test_size=0.2, random_state=0)
    y_train =  np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    x_train = vr.fit_transform(x_train)
    mask = vr.get_support()
    x_test = np.array(x_test)[:,mask]

    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
 
    # Sample hyper parameters
    C = trial.suggest_float('C', 1e-10, 1)
    kernel = trial.suggest_categorical('kernel',['poly','rbf','sigmoid'])
    degree = trial.suggest_int('degree',1, 50)
    
    clf = SVC(C=C,kernel=kernel,degree=degree, random_state=0, max_iter=3000)
    score = np.mean(cross_val_score(clf,x_train,y_train,scoring='balanced_accuracy',cv=10))
    return score

data = pd.read_csv("./Dry_Bean.csv")
n_samples=data.shape[0]
n_features = data.shape[1] - 1
x = data.drop('Class',axis=1)
y = data[["Class"]]

study = optuna.create_study(direction='maximize')
study.optimize(objective_svm, n_trials=50)
