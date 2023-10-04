from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import pandas as pd
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import joblib
import pydotplus

path='input.csv'
df=pd.read_csv(path)

from sklearn import utils
data=utils.shuffle(df)

Y=data['label'].values
X=data.drop('label', axis=1).values

scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

train_X, test_X, train_Y, test_Y=train_test_split(X, Y, test_size=0.2, random_state=0)
# clf = RandomForestClassifier(n_estimators=3, criterion="entropy", min_samples_leaf=3, max_depth=7)

clf = RandomForestClassifier(n_estimators=7, criterion="gini", min_samples_leaf=3, max_depth=5)
kfold = KFold(n_splits=5)

# 训练
clf.fit(train_X, train_Y)

feat_labels = df.columns[1:]
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
# for f in range(train_X.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# 预测并评估
predict_Y = clf.predict(test_X)
print(len(test_X),len(predict_Y))
print(f"准确率 = {accuracy_score(y_pred=predict_Y, y_true=test_Y)}")
print(f"精度 = {precision_score(y_true=test_Y, y_pred=predict_Y,average='micro')}")
print(f"召回率 = {recall_score(y_true=test_Y, y_pred=predict_Y,average='micro')}")
print(f"f1 = {f1_score(y_true=test_Y, y_pred=predict_Y,average='micro')}")
t = classification_report(y_pred=predict_Y, y_true=test_Y)
print(t)

joblib.dump(clf,"model.pkl")

