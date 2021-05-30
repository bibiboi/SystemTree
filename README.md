# SystemTree
系统发生树构建（非加权分组平均法python实现）





<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 算法描述

<font color=#999AAA >分子进化与系统发生：计算分子进化——利用算法在分子水平上构建物种的进化树。这里说的分子水平是指DNA、RNA、以及蛋白质序列。

*1. ***非加权分组平均法：****
对于某几条不同的序列，找出距离最小的一对序列（两条序列之间的距离为两条序列相同位置上不同的碱基总数），将其合并聚集，形成一个分支，聚集后将两条序列看成一个整体，再分别计算与其他序列之间的距离，直到所有序列归为一类，系统发生树构建完成，树枝的长短直接反应了它们与共同祖先的距离。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011222833748.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
                                                                         
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011222857212.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
                                                                         
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011222907871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
                                                                         
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011222918743.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
                                                                         
***2. 邻接法：***
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201027021905228.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
                                                                         
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201027021954368.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201027021904765.png#pic_center)

</font>

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">


# 设计思想

 ***- 非加权分组平均法：***
  1.  读取输入的序列并将进行命名
     2. 构建距离矩阵（使用pandas库的dataframe进行存储）
     3. 使用循环逐步更新距离矩阵，直到系统发生树构建完成（即距离矩阵行只剩一行），更新距离矩阵分为以下四种情况：

 		
|  | A| B| C| D|
|--|--|--|--|--|
|B |① | |
|C| |③|
|D|
|E|④|| |②|



①最小距离在AB处：列A与列B进行相加计算得聚类AB后的距离，赋值给列A，删除行B及列B得到最新距离矩阵如下

|  | A B|   C|  D  |
|--|--|--|--|
|C|||
|D|
|E||

②最小距离在DE处：行C与行D进行相加计算得聚类CD后的距离，赋值给行E，删除行D及列D得到最新距离矩阵如下

 |  | A| B|   C| 
|--|--|--|--|
|B|||
|C|
|DE||

③最小距离在BC处：列B与列C进行相加计算得聚类BC后的距离，赋值给列B，删除行C；行B与行C进行相加计算得聚类BC后的距离，赋值给行B，删除行C，得到最新距离矩阵如下

 |  | A| BC|   D| 
|--|--|--|--|
|BC|||
|D|
|E||

④最小距离在AE处：A列B、C、D行分别与E行B、C、D列相加后减半，删除行E，删除最后一列D，得到最新距离矩阵如下

 |  | AE| B|   C| 
|--|--|--|--|
|B|||
|C|
|D||

 -  画出系统发生树，根据各分支的关系及聚类之间最小距离设置节点坐标，并使用matplotlib的plot函数从根节点往分支方向依次画出系统发生树，各节点的纵坐标为距离值

***- 邻接法：***
 -  读取输入的序列并将进行命名
 - 构建距离矩阵（使用pandas库的dataframe进行存储）
 - 计算初始初始净分歧度和最小速率校正距离
 - 使用循环逐步更新距离矩阵，情况与非加权分组平均法类似，但计算方法略有不同，还需计算净分歧度r及最小速率校正距离
 
# 源代码及运行成果
## 1.运行成果

 - *非加权分组平均法*

<font color=#999AAA >运行截图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011231457964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011231509720.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201011231518962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)

 - *邻接法*
  
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201027024116921.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)


## 2.源代码

<font color=#999AAA >**非加权分组平均法** 代码如下：





```python
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
```

<font color=#999AAA >**邻接法** 代码如下：

```python
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
    return minIndexName, minColNameumns

#计算r
def createR(columns,index,distance):
    r = {}
    for s in columns+index:
        if s == columns[0]:#第一个序列求和
            r[s] = distance[s].sum()
        elif s == index[-1]:#最后一个序列求和
            r[s] = distance.loc[s].sum()
        else:#中间序列求和
            r[s] = distance[s].sum() + distance.at[s,columns[0]]
    return r
#计算M
def createM(columns,index,M,distance,r,length):
    for col in columns:
        for ind in index:
            if col != ind:
                M.at[ind,col] = distance.at[ind,col] - (r[col] + r[ind])/(length-2)


#输入序列命名
LetterIndex = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#输入序列
Sequences = []
# Sequences = ['tagg','tacg','aagc','agcc']

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
#存储父节点与子节点的关系
LeafDict = {}
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
    print("distance:")
    print(distance)
    # 计算r净化距离
    r = createR(columns, index, distance)
    # 计算最小速率校正矩阵
    M = distance.copy(deep=True)
    createM(columns, index, M, distance, r, length)
    # print("M:" )
    # print(M)
    #获取表格中最小值的行名、列名
    minIndexName,minColName  = getMinName(M)
    #连接行名及列名
    newSeName = minColName + minIndexName
    num = distance.at[minIndexName,minColName]
    LeafDict[newSeName] = [minColName,minIndexName]
    order.append(newSeName)

    #更新序列合并后的距离矩阵
    if len(distance) > 1:
        if minIndexName in columns and minColName not in index :
            distance.drop(minIndexName,axis=0,inplace=True)# 删除多余行
            distance[minColName] = (distance[minColName] + distance[minIndexName] - num)/2# 求得两序列连接后的距离矩阵
            distance.rename(columns={minColName: newSeName}, inplace=True)# 改列名
            distance.drop(minIndexName, axis=1, inplace=True)# 删除多余列
            columns = list(distance)# 更新列名
            index = distance._stat_axis.values.tolist()  # 更新行名
        elif minColName in index and minIndexName not in columns:
            distance.drop(minColName, axis=1, inplace=True)  # 删除多余列
            distance.loc[minIndexName] = (distance.loc[minIndexName] + distance.loc[minColName] - num) / 2 # 算新距离矩阵
            distance.rename(index={minIndexName: newSeName}, inplace=True)  # 改行名
            distance.drop(minColName, axis=0, inplace=True)  # 删除多余行
            columns = list(distance)  # 更新列名
            index = distance._stat_axis.values.tolist() # 更新行名
        elif minIndexName in columns and minColName in index:# 若行名列名都出现在行索引及列索引集合中
            distance[minColName] = (distance[minColName] + distance[minIndexName] -num) / 2
            distance.rename(columns={minColName: newSeName}, inplace=True)  # 改列名
            distance.drop(minIndexName, axis=1, inplace=True)  # 删除minIndexName列
            distance.loc[minColName] = (distance.loc[minIndexName] + distance.loc[minColName] -num) / 2  # 算新距离矩阵
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
                distance.at[id,minColName] = (distance.at[id,minColName] + distance.at[minIndexName,id] - num)/2
            distance.rename(columns={minColName: newSeName}, inplace=True)  # 改列名
            distance.drop(minIndexName, axis=0, inplace=True)  # 删除minIndexName行
            distance.drop(list(distance)[-1], axis=1, inplace=True)  # 删除最后一列
            #更新行名及列名
            columns = list(distance)
            index = distance._stat_axis.values.tolist()
    else:
        break
print("============================")
print("聚类如下:")
print(LeafDict)
```

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 遇到的问题及总结
## 问题

 1. 输入序列的读取与命名  *解决方法*-使用含有26个字母的字符串进行依次转换命名（待完善）
 2. 思维误区1：更新距离矩阵考虑情况过于简单，只考虑到情况①②，导致调试报错 *解决方法*-逐步调试完善情况
 3. 思维误区2：画树时从叶子节点往根节点方向画导致叶子节点无法改变，造成画树时树枝交叉缠绕 *解决方法*-另外创建数据结构存储画树所需节点信息，从根节点往叶子方向画（待完善）

## 总结
  系统发生树的距离矩阵我使用了pandas库的dataframe进行存储，相对于numpy的二维数组，dataframe表格在对行列进行操作时更加方便，不需要使用循环而可以直接相加减，实现距离矩阵的更新时简化了代码，但对于是否提升时间复杂度还有待学习，需要了解pandas的dataframe的内部实现，暂时没有查到资料（了解到dataframe比较适用于操作表格）。
  
最初实现算法时由于头脑简单忽略了很多矩阵更新的情况导致后来代码大改写，严重影响到整个作业的进度和心态。画树的过程中也遇到了很大的问题，因为最初头脑简单考虑不周导致只能画出某个特定情况下的树，由于对matplotlib的不熟悉可能也没有选择最佳的画图函数，第一次画树失败后卡了很久甚至一度想放弃画图，最后牺牲了空间开销，转变思维方向画出了树的结构，但树的形状属实还有待美化。

在实现邻接法构建树的过程中由于时间原因没有完成树的构建，只是输出了距离矩阵和分类。

归根到底还是对算法理解不到位以及对python的pandas和画图工具库不熟悉，导致实现非加权分组平均法时耗费了过多时间。总的来说虽然使用非加权分组平均法构建出了系统发生树，但代码中有很多不成熟的处理方式，且邻接法画树的步骤还有待完成，还十分需要改进和完善。
