#!/usr/bin/python
# -*- coding: utf-8 -*-

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # 将数据点的ý轴数据与拟合直线的预测数据预测进行比较，差值（也称为残差）大的就判定。异常为值注意残差值的英文需要先对原始数据进行拟合之后才能进行计算的，残差 = | 预测值 - 实际值 |。
    #
    # 定义一个函数，去除数据中残差值前10％的数据

    cleaned_data = []

    # 生成残差列表
    err = []
    for i in range(len(predictions)):
        err.append(abs(net_worths[i][0] - predictions[i][0]))

    # 剔除异常数据,在此处一定要注意不能修改原数据,即不能修改传进来的列表参数
    stay_index = list(range(len(predictions)))  # 保留列表
    for i in range(int(0.1 * len(predictions))):
        max_val = max(err)
        max_index = err.index(max_val)
        del stay_index[max_index], err[max_index]

    for index in stay_index:
        cleaned_data.append((ages[index][0], net_worths[index][0], abs(net_worths[index][0] - predictions[index][0])))

    return cleaned_data
