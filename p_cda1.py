# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:16:11 2020

@author: mlt
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import svm
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report 
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier



train=pd.read_csv('D://操作练习/案例题资料/第一套模拟题案例分析题数据/cs-training.csv')
test=pd.read_csv('D://操作练习/案例题资料/第一套模拟题案例分析题数据/cs-test.csv')
test=test.iloc[:,:-1:]

####preprocess
train['MonthlyIncome']=train['MonthlyIncome'].fillna(train['MonthlyIncome'].mean())
train['NumberOfDependents']=train['NumberOfDependents'].fillna(train['NumberOfDependents'].mean())
test['MonthlyIncome']=test['MonthlyIncome'].fillna(train['MonthlyIncome'].mean())
test['NumberOfDependents']=test['NumberOfDependents'].fillna(train['NumberOfDependents'].mean())
train_2=train.iloc[:,1:-1:]
test_2=test.iloc[:,1::]

#for column in list(train_2.columns):
#    t5_val=float(np.percentile(train_2[column],[5]))
#    t95_val=float(np.percentile(train_2[column],[95]))
#    train_2.loc[train_2[column]<t5_val,column]=t5_val
#    train_2.loc[train_2[column]>t95_val,column]=t95_val
#    test_2.loc[test_2[column]<t5_val,column]=t5_val
#    test_2.loc[test_2[column]>t95_val,column]=t95_val
    
for column in list(train_2.columns):
    tstd_val1=np.mean(train_2[column])-3*np.std(train_2[column])
    tstd_val2=np.mean(train_2[column])+3*np.std(train_2[column])
    train_2.loc[train_2[column]<tstd_val1,column]=tstd_val1
    train_2.loc[train_2[column]>tstd_val2,column]=tstd_val2
    test_2.loc[test_2[column]<tstd_val1,column]=tstd_val1
    test_2.loc[test_2[column]>tstd_val2,column]=tstd_val2
    

    
train.iloc[:,1:-1:]=train_2
test.iloc[:,1::]=test_2


train_x=train.iloc[:,1:-1:]
train_y=train.iloc[:,-1]
test_id=test.iloc[:,0]
test_x=test.iloc[:,1::]
test_yy=pd.read_csv('D://操作练习/案例题资料/第一套模拟题案例分析题数据/cs-test v2.csv')
test_y=test_yy.iloc[:,-1]

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_sample(train_x, train_y)

train_y=y_resampled
ss=StandardScaler()
train_x=ss.fit_transform(X_resampled)
test_x=ss.transform(test_x)

#####modeling

#### svm 太费时
##clf=make_pipeline(StandardScaler(),svm.SVC())
##clf=svm.SVC(C=1.0, kernel='linear', gamma=20, decision_function_shape='ovo')
##C=1.0,kernel='poly' /'rbf'
#clf.fit(train_x,train_y)
#test_pre=clf.predict(test_x)
#print(clf.score(test_x,test_y))
#res3=pd.concat([test_id,pd.Series(test_pre,name='SeriousDlqin2yrs')],axis=1)
#res3.to_csv('pred.csv',index=False,header=True)


###### AdaBoostClassifier
#clf=AdaBoostClassifier(n_estimators=100)
#clf.fit(train_x,train_y)
#test_pre=clf.predict(test_x)
#print("F1 Score: %f" %metrics.f1_score(test_y,test_pre))
#print(classification_report(test_y,test_pre,target_names=['0','1']))

####
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.07,
#                                 max_depth=7, random_state=0)
#clf.fit(train_x,train_y)
#test_pre=clf.predict(test_x)
#print("F1 Score: %f" %metrics.f1_score(test_y,test_pre))
#print(classification_report(test_y,test_pre,target_names=['0','1']))

#res3=pd.concat([test_id,pd.Series(test_pre,name='SeriousDlqin2yrs')],axis=1)
#res3.to_csv('pred.csv',index=False,header=True)


####XGB 
clf=XGBClassifier()
clf.fit(train_x,train_y)
test_pre=clf.predict(test_x)
print("F1 Score: %f" %metrics.f1_score(test_y,test_pre))
print(classification_report(test_y,test_pre,target_names=['0','1']))

res=pd.concat([test_id,pd.Series(test_pre)],axis=1)
res.columns=['CustomerId','SeriousDlqin2yrs']
res.to_csv('./pred_3.csv',index=False,header=True)










