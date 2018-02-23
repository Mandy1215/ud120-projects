#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
#特征数量和过拟合
from sklearn.metrics import accuracy_score
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pre = clf.predict(features_test,labels_test)
print(accuracy_score(pre,labels_test))  #准确率

pred = clf.predict(features_train)
print(len(pred))    ##根据初始代码，有多少训练点？

#识别强大的特征
#最重要特征的重要性是什么？该特征的数字是多少？
imp = clf.feature_importances_
print imp.max()
print imp.argmax()


#使用 TfIdf 获得最重要的单词
key_index = 0
for index, feature in enumerate(clf.feature_importances_):
    if feature > 0.2:
        key_index = index
        print(index, feature)
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
# vectorizer.fit_transform(word_data)
# vectorizer_list = vectorizer.get_feature_names()
# print vectorizer_list[key_index]
word_bags = vectorizer.get_feature_names()
print word_bags[key_index]


#删除、重复  ##出现新的异常值“cgermannsf”
