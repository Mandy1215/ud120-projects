#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

#读入数据（以字典形式）并将之转换为适合 sklearn 的 numpy 数组。
# 由于从字典中提取出了两个特征（“工资”和“奖金”），
# 得出的 numpy 数组维度将是 N x 2，其中 N 是数据点数，2 是特征数。
### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
#print (data_dict)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
# 绘制散点图  matplotlib.pyplot 模块来绘制图形
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


#找出异常值数据的具体信息
persons=[]
salary=[]
bonus=[]

for person in data_dict:
    persons.append(person)
    salary.append(data_dict[person]['salary'])
    bonus.append(data_dict[person]['bonus'])

#处理缺省值
for i in range(len(salary)):
    if(salary[i]=='NaN'):
        salary[i]=0

max_index=salary.index(max(salary))         #此处也可依照'bonus'特征查找最大值索引
key=persons[max_index]
print(key,data_dict[key])

#删除异常值
data_dict.pop( 'TOTAL', 0 )

#print ('PHILLIP ALLEN K' , data_dict['PHILLIP ALLEN K'])

print (data_dict["LAY KENNETH L"]["salary"])
print (data_dict["SKILLING JEFFREY K"]["salary"])
print (data_dict["FASTOW ANDREW S"]["salary"])