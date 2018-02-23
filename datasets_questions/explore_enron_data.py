#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#数据集中有多少数据点（人）？
print(len(enron_data))

##对于每个人，有多少个特征可用？
print (len(enron_data["SKILLING JEFFREY K"]))

#E+F 数据集中有多少 POI？

#在安然数据中查找 POI(也就是说，计算 data[person_name]["poi"]==1 时，字典中条目的数量。)
print (len(dict ((key,value) for key,value in enron_data.items() if value["poi"] == 1)))

#存在多少 POI？
poi_reader =open("../final_project/poi_names.txt", "r")
fr=poi_reader.readlines()
print(len(fr[2:]))
poi_reader.close()

#和任何字典一样，个人特征可以通过以下代码访问：
#enron_data["LASTNAME FIRSTNAME"]["feature_name"]
#或者
#enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]
#James Prentice 名下的股票总值是多少？
print (enron_data["PRENTICE JAMES"]["total_stock_value"])


#和任何字典的字典一样，个人/特征可以这样被访问：
#enron_data["LASTNAME FIRSTNAME"]["feature_name"]
#我们有多少来自 Wesley Colwell 的发给嫌疑人的电子邮件？
print (enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

#Jeffrey Skilling 行使的股票期权价值是多少？
print (enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

#这三个人（Lay、Skilling 和 Fastow）当中，谁拿回家的钱最多（“total_payments”特征的最大值）？
#这个人得到了多少钱？
print (enron_data["LAY KENNETH L"]["total_payments"])
print (enron_data["SKILLING JEFFREY K"]["total_payments"])
print (enron_data["FASTOW ANDREW S"]["total_payments"])


people = ["LAY KENNETH L" ,"SKILLING JEFFREY K", "FASTOW ANDREW S" ]
money = 0
person = 0
for i in people:
    if money < enron_data[i]["total_payments"]:
        money = enron_data[i]["total_payments"]
        person = i
print (person, money)

#此数据集中有多少雇员有量化的工资？已知的邮箱地址是否可用？
count_salary , count_email = 0,0
for i in enron_data:
    if enron_data[i]["salary"] != "NaN":
        count_salary += 1
    if enron_data[i]["email_address"] != "NaN":
        count_email += 1
print (count_salary,count_email)

#（当前的）E+F 数据集中有多少人的薪酬总额被设置了“NaN”？数据集中这些人的比例占多少？
count_totalpay = 0
for i in enron_data:
    if enron_data[i]["total_payments"] == "NaN":
        count_totalpay += 1
print (count_totalpay , count_totalpay*1.0/146)

#E+F 数据集中有多少 POI 的薪酬总额被设置了“NaN”？这些 POI 占多少比例？
totalpoi_count = 0
poi_num = 0
for i in enron_data:
    if enron_data[i]["poi"] == True:
        poi_num += 1
        if enron_data[i]["total_payments"] == "NaN":
            totalpoi_count += 1
print (poi_num, totalpoi_count,totalpoi_count*1.0/poi_num)

