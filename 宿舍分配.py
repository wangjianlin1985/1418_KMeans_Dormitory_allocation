#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from matplotlib.figure import SubplotParams
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']##中文乱码问题！
plt.rcParams['font.sans-serif']=['Microsoft YaHei']##中文乱码问题！
plt.rcParams['axes.unicode_minus']=False#横坐标负号显示问题！


# In[2]:


'''
关于读取数据，
第一种方法：按照您们自己的方法读取数据
第二中方法：在D盘，新建文件夹，改名为 me，把数据集放到该文件中，代码不用改动，可以直接运行

'''
path = 'D:\\项目系统开发区\\2019-2023年定做\\洋子2023年新建文件夹\\聚类算法在高校宿舍分配中的应用\\'#请您正确填写自己数据路径，只是路径、路径，不要输入文件名！！！

df_name1 = '宿舍画像数据.xlsx'  # 该行不需要改动
data = pd.read_excel(path + df_name1)


# In[3]:


data.head()


# In[4]:


#哑变量处理
data['性别'][data['性别'] == '男'] = 1
data['性别'][data['性别'] == '女'] = 2

data['区域'][data['区域'] == '华中'] = 1
data['区域'][data['区域'] == '华南'] = 2
data['区域'][data['区域'] == '华北'] = 3
data['区域'][data['区域'] == '东北'] = 4
data['区域'][data['区域'] == '华东'] = 5
data['区域'][data['区域'] == '西南'] = 6

data['身高'][data['身高'] == '150-160'] = 1
data['身高'][data['身高'] == '160-170'] = 2
data['身高'][data['身高'] == '170-180'] = 3
data['身高'][data['身高'] == '180-190'] = 4

data['作息时间'][data['作息时间'] == '12-0'] = 1
data['作息时间'][data['作息时间'] == '10-11'] = 2
data['作息时间'][data['作息时间'] == '11-12'] = 3

data['年度旅游频次'][data['年度旅游频次'] == '0-3'] = 1
data['年度旅游频次'][data['年度旅游频次'] == '3-5'] = 2
data['年度旅游频次'][data['年度旅游频次'] == '5-7'] = 3

data['饮食口味'][data['饮食口味'] == '甜'] = 1
data['饮食口味'][data['饮食口味'] == '辣'] = 2


# In[5]:


data.head()


# In[6]:


df = (data - data.min()) / (data.max() - data.min())  # 数据归一
df


# In[7]:





choose_k(df)
# 根据手肘法，轮廓系数和CH值，K取4


# In[9]:


k = 4
kmeans=KMeans(n_clusters=k)
kmeans.fit(df)
r1 = pd.DataFrame(kmeans.cluster_centers_)  # 找出聚类中心
r2 = pd.Series(kmeans.labels_).value_counts()
r3 = round((r2 / df.shape[0]), 5) * 100
r = pd.concat([r1,r2,r3], axis=1)
r.columns = ['聚类中心' + str(i + 1) for i in range(r1.shape[1])] + ['分层各类别数目'] + ['占比总数目（%）']  # 自定义列台头
rr = pd.concat([data, pd.Series(kmeans.labels_,index=df.index.to_list())], axis=1)
rr = rr.rename(columns={0:'聚类类别'})

rr.to_csv('最终聚类结果展示.csv',encoding='utf_8_sig')
r.to_csv('聚类中心及占比展示.csv',encoding='utf_8_sig')

print("{:*^60}".format("最终聚类结果展示"))
print(rr)

print("{:*^60}".format("表聚类中心及占比展示"))
print(r)


# In[10]:

 

# 使用TSNE 进行数据降维并展示聚类结果
tsne = TSNE()   # 进行降维
tsne.fit_transform(df)
tsn = pd.DataFrame(tsne.embedding_, index= df.index)
color_sty = ['c*', 'r*', 'bo','o']
for i in range(4):  # 按照类别循环画图
    d = tsn[rr['聚类类别'] == i ]
    plt.plot(d[0], d[1], color_sty[i], label='C' + str(i+1))
plt.legend()
plt.show()


# In[12]:


df.columns.tolist()


# In[13]:


def cluster_radar(df_k):
    '''
    聚类雷达图
    
    '''

    # 定义k-means模型并聚类
    kmeans = KMeans(n_clusters=4)   # 聚成4类
    kmeans.fit(df_k)

    # 获取聚类中心
    centers = kmeans.cluster_centers_
    min_max_model = MinMaxScaler()
    centers = min_max_model.fit_transform(centers)  # 标准化的数据使其在0-1之间

    labels = df_k.columns.tolist()  # 获取列表

    # 绘制雷达图
    n = centers.shape[1]  # 特征数
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)  # 计算各个区间的角度
    angles = np.concatenate((angles, [angles[0]]))  # 建立相同首尾字段以便于闭合
    labels = np.concatenate((labels,[labels[0]]))

    colors = ['b', 'y', 'r', 'g']
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    for i in range(centers.shape[0]):
        data = np.concatenate((centers[i], [centers[i][0]]))  # 闭合
        ax.plot(angles, data, color=colors[i],label='聚类类别{}'.format(i+1))
    
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(90)
    ax.set_rlim(-0.2, 1.2)  # 展示数据范围
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0, 1.1))
    plt.show()
    
cluster_radar(df)    


# In[14]:


def plot_data(ax, x_label, y_label, x_data, y_data):
    ax.bar(x_data, y_data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


# In[15]:


data.head()
data.info()


# In[16]:


fig, axs = plt.subplots(4,4)
fig.set_size_inches(30,30)

print('*'*20, '聚类结果分析画图', '*'*20)
for i in range(k):
    xd = data[rr['聚类类别'] == i]
    xd1 = xd['性别'].value_counts()
    xd2 = xd['年龄'].value_counts()
    xd3 = xd['区域'].value_counts()
    xd4 = xd['身高'].value_counts()
    
    plot_data(axs[i,0], "性别", "数量", xd1.index.tolist(), xd1.values.tolist())
    plot_data(axs[i,1], "年龄", "数量", xd2.index.tolist(), xd2.values.tolist())
    plot_data(axs[i,2], "区域", "数量", xd3.index.tolist(), xd3.values.tolist())
    plot_data(axs[i,3], "身高", "数量", xd4.index.tolist(), xd4.values.tolist())
    
plt.show()


# In[17]:



# In[ ]:





# In[ ]:





# In[ ]:




