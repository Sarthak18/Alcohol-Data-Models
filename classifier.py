import pandas as pd
import numpy as np

data_mat = pd.read_csv("student-mat.csv", sep=";")
data_por = pd.read_csv("student-por.csv", sep=";")

data = data_mat.append(data_por, ignore_index=True)

data = data.drop(['school', 'address', 'reason'], axis=1)

data_one_hot_encoded = pd.get_dummies(data)

walc = data['Walc']
dalc = data['Dalc']

data_one_hot_encoded = data_one_hot_encoded.drop(['Walc', 'Dalc'], axis=1)

print 'Length of data collected from files : ', len(data_one_hot_encoded)

from sklearn.cross_validation import train_test_split
X_train_dalc, X_test_dalc, y_train_dalc, y_test_dalc = train_test_split(data_one_hot_encoded, dalc, test_size=0.1, random_state=42)

from sklearn import linear_model

from sklearn import grid_search

print 'Intializing paramters for grid search...'

parameters = {'C':[0.1,0.5,1.0,5.0,10.0,20.0,50.0,100.0,500.0,1000.0], 
				'random_state': [i*5 for i in range(1,20)]}


from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
scorer = make_scorer(accuracy_score)

print 'Intializing Logisic Regression classifier...'
clf = linear_model.LogisticRegression()

print 'Applying Logistic Regression to predict dalc...'
grid_obj_dalc = grid_search.GridSearchCV(clf,parameters,scoring=scorer)
grid_fit_dalc = grid_obj_dalc.fit(X_train_dalc, y_train_dalc)
best_clf_dalc = grid_fit_dalc.best_estimator_

pred_dalc = best_clf_dalc.predict(X_test_dalc)

acc_score_dalc = accuracy_score(pred_dalc, y_test_dalc)

print 'Accuracy score for dalc ', acc_score_dalc 

X_train_walc, X_test_walc, y_train_walc, y_test_walc = train_test_split(data_one_hot_encoded, walc, test_size=0.1, random_state=42)

print 'Applying Logistic Regression to predict walc...'
grid_obj_walc = grid_search.GridSearchCV(clf,parameters,scoring=scorer)
grid_fit_walc = grid_obj_walc.fit(X_train_walc, y_train_walc)
best_clf_walc = grid_fit_walc.best_estimator_

pred_walc = best_clf_dalc.predict(X_test_walc)

acc_score_walc = accuracy_score(pred_walc, y_test_walc)

print 'Accuracy score for walc', acc_score_walc 