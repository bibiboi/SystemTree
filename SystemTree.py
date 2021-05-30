#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pandas import DataFrame
from pylab import *
from matplotlib import pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']#解决中文乱码

#计算两序列之间的距离
def CalDistance(str1,str2):
    dis = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
           dis += 1
    return dis

#获取dataframe中最小值的行名列名
def getMinName(df):
    #最小值所在列
    minColNameumns = df.min().idxmin()
    #最小值
    Min = df.min().min()
    #最小值所在行
    minIndexName = ''.join(df[(df[minColNameumns] == Min)].index.tolist()[0])#转换为list再转换为string
    return minIndexName, minColNameumns,Min

#输入序列命名
LetterIndex = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#输入序列
Sequences = []
print("请输入序列（每条序列以换行符相隔）：")
for line in iter(input, ''):
    Sequences.append(line.replace(',',''))
length = len(Sequences)
#创建列名序列
columns = list(LetterIndex)[:length-1]
#创建行名序列
index = list(LetterIndex)[1:length]
#创建距离矩阵
distance = DataFrame(np.zeros((length-1,length-1)),index = index,columns = columns)
#存储系统发生树的节点坐标
PointList = {}
#存储父节点与子节点的关系
LeafDict = {}
#画树所需节点集合
xPointList = []
yPointList = []
MinDistance = {}
order = []

print("输入序列如下   ")
for i in range(length):
    print("序列",LetterIndex[i],":",Sequences[i])
#计算距离矩阵
for i in range(length - 1):
    for j in range(length - 1):
        if columns[i] != index[j] :
            # 序列名字转换并计算各序列的距离
            distance.loc[index[j],columns[i]] = CalDistance(Sequences[LetterIndex.index(columns[i])],
                                                            Sequences[LetterIndex.index(index[j])])
        else:
            distance.loc[index[j], columns[i]] = np.NaN
#构建距离矩阵
while len(distance) >= 1:#判断是否构建为最终的距离矩阵len(distance) >= 1
    print("============================")
    print(distance)
    #获取表格中最小值的行名、列名
    minIndexName,minColName,Min  = getMinName(distance)
    #连接行名及列名
    newSeName = minColName + minIndexName
    #存储系统发生树中父子节点关系
    LeafDict[newSeName] = [minColName,minIndexName]
    #存储最小距离
    MinDistance[newSeName] = Min/2
    #存储父节点构建顺序
    order.append(newSeName)
    # # 判断叶子节点并计算叶子节点坐标
    # if minColName in list(LetterIndex)[:length]:
    #     PointList[minColName] = (0, LeafNum / length)
    #     LeafNum += 1
    # if minIndexName in list(LetterIndex)[:length]:
    #     PointList[minIndexName] = (0, LeafNum / length)
    #     LeafNum += 1
    # # 计算根节点坐标，x为叶子节点两序列距离的一半，y为叶子节点y值的一半
    # y = (PointList[LeafDict[newSeName][0]][1] +
    #      PointList[LeafDict[newSeName][1]][1]) / 2
    # PointList[newSeName] = (Min / 2, y)

    #更新序列合并后的距离矩阵
    if len(distance) > 1:
        if minIndexName in columns and minColName not in index :
            distance.drop(minIndexName,axis=0,inplace=True)# 删除多余行
            distance[minColName] = (distance[minColName] + distance[minIndexName])/2# 求得两序列连接后的距离矩阵
            distance.rename(columns={minColName: newSeName}, inplace=True)# 改列名
            distance.drop(minIndexName, axis=1, inplace=True)# 删除多余列
            columns = list(distance)# 更新列名
        elif minColName in index and minIndexName not in columns:
            distance.drop(minColName, axis=1, inplace=True)  # 删除多余列
            distance.loc[minIndexName] = (distance.loc[minIndexName] + distance.loc[minColName]) / 2 # 算新距离矩阵
            distance.rename(index={minIndexName: newSeName}, inplace=True)  # 改行名
            distance.drop(minColName, axis=0, inplace=True)  # 删除多余行
            index = distance._stat_axis.values.tolist() # 更新行名
        elif minIndexName in columns and minColName in index:# 若行名列名都出现在行索引及列索引集合中
            distance[minColName] = (distance[minColName] + distance[minIndexName]) / 2
            distance.rename(columns={minColName: newSeName}, inplace=True)  # 改列名
            distance.drop(minIndexName, axis=1, inplace=True)  # 删除minIndexName列
            distance.loc[minColName] = (distance.loc[minIndexName] + distance.loc[minColName]) / 2  # 算新距离矩阵
            distance.rename(index={minColName: newSeName}, inplace=True)  # 改行名
            distance.drop(minIndexName, axis=0, inplace=True)  # 删除minColName行
            distance.at[newSeName,newSeName] = np.NaN
            # 更新行名及列名
            columns = list(distance)
            index = distance._stat_axis.values.tolist()
        else:
            #获取行名列表
            newindex = distance._stat_axis.values.tolist()
            newindex.remove(minIndexName)
            #计算新距离矩阵
            for id in newindex:
                distance.at[id,minColName] = (distance.at[id,minColName] + distance.at[minIndexName,id])/2
            distance.rename(columns={minColName: newSeName}, inplace=True)  # 改列名
            distance.drop(minIndexName, axis=0, inplace=True)  # 删除minIndexName行
            distance.drop(list(distance)[-1], axis=1, inplace=True)  # 删除最后一列
            #更新行名及列名
            columns = list(distance)
            index = distance._stat_axis.values.tolist()
    else:
        break

# 设置树的节点坐标
for i in range(len(order)-1,-1,-1):
    root = order[i]
    if i == len(order) - 1:
        interval = 0.4
        maxDistance = MinDistance[root]
        PointList[root] = (interval,0)
    interval /= 2
    leaf1 = LeafDict[root][0]
    leaf2 = LeafDict[root][1]
    if leaf1 not in order:
        PointList[leaf1] = (PointList[root][0] + interval, maxDistance)
    else:
        PointList[leaf1] = (PointList[root][0] + interval,maxDistance - MinDistance[leaf1])
    if leaf2 not in order:
        PointList[leaf2] = (PointList[root][0] - interval, maxDistance)
    else:
        PointList[leaf2] = (PointList[root][0] - interval, maxDistance - MinDistance[leaf2])

    # 为画树构建节点列表
    xPointList.append([PointList[leaf1][0], PointList[root][0]])
    xPointList.append([PointList[leaf2][0], PointList[root][0]])
    yPointList.append([PointList[leaf1][1], PointList[root][1]])
    yPointList.append([PointList[leaf2][1], PointList[root][1]])

# 画树
fig=plt.figure()
ax=fig.add_subplot(111)
for i in range(len(xPointList)):
    plt.plot(xPointList[i],yPointList[i],color = 'black',alpha = 0.5)
    plt.scatter(xPointList[i],yPointList[i],color = 'black',alpha=0.8)
# 标明注释
for k,v in PointList.items():
    if k in list(LetterIndex)[:length]:
        strName = k +":" + Sequences[LetterIndex.index(k)]
        plt.annotate(s = strName,xy=v,xytext=(v[0]-10,v[1]+10) ,rotation = 45,textcoords = 'offset points',alpha = 0.5)
plt.ylabel("距离")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()