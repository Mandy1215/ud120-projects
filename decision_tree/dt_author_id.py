#!/usr/bin/python
# -*- coding: utf-8 -*-
""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t = time()
clf.fit(features_train,labels_train)
print "time:",round(time()-t, 3),"s"
pred = clf.predict(features_test)

acc_min_samples_split_40 = clf.score(features_test,labels_test)
print acc_min_samples_split_40
print len(features_train[0])



from sklearn.metrics import accuracy_score
acc_40 = accuracy_score(pred, labels_test)
print acc_40
#########################################################


