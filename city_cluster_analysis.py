#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates


# In[2]:


data1 = pd.read_excel(r'E:\城市活力\新城市活力\原数据表/新城市活力归一化.xlsx')
filled_df = data1.copy()
filled_df.head(10)


# In[4]:


from pyecharts.charts import Bar
from pyecharts import options as opts
bar = (
    Bar()
    .add_xaxis(filled_df['地市名称'].tolist())
    .add_yaxis("非老龄化程度", filled_df['非老龄化程度'].tolist(), color="#1565C0")
    .add_yaxis("高学历人口占比", filled_df['高学历人口占比'].tolist(), color="#80CBC4")
    .set_global_opts(title_opts=opts.TitleOpts(title="人口结构", subtitle="学历和年龄占比"),
                     yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size = 15)),
                    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.TextStyleOpts(font_size = 12)))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False),
                     markpoint_opts=opts.MarkPointOpts(
         data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]), 
         markline_opts=opts.MarkLineOpts(
            data=[
                opts.MarkLineItem(type_="average", name="平均值"),
            ])
 )
.reversal_axis()
)
bar.render_notebook()


# # 空值填充

# In[5]:


from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

filled_df.set_index(['地市名称'], inplace=True)
updated_df6=filled_df.copy()
scaler = MinMaxScaler()
updated_df6 = pd.DataFrame(scaler.fit_transform(updated_df6), columns = updated_df6.columns)
updated_df6.head(10)
#k = 9为最佳

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=11)
updated_df7 = pd.DataFrame(imputer.fit_transform(updated_df6),columns = updated_df6.columns)
updated_df7.tail(10)


# In[7]:


#k值选择对比 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

def optimize_k(data, target):
    errors = []
    for k in range(1, 20, 2):
        scaler = MinMaxScaler()
        sclaed_df = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
        imputer = KNNImputer(n_neighbors=k)
        imputed_df = pd.DataFrame(imputer.fit_transform(data),columns = data.columns)
        
        X = imputed_df.drop(target, axis=1)
        y = imputed_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error = rmse(y_test, preds)
        errors.append({'K': k, 'RMSE': error})
        
    return errors


# In[20]:


import pandas as pd
import numpy as np

temp = updated_df6
k_errors = optimize_k(data=temp, target='营商环境指数')
print(k_errors)
#k=5时R


# In[32]:


#填充空值
filled_df['营商环境指数'] = filled_df['营商环境指数'].fillna(11.00091)


# #相关性分析

# In[47]:


get_ipython().system('wget -O /usr/share/fonts/truetype/liberation/simhei.ttf "https://www.wfonts.com/download/data/2014/06/01/simhei/chinese.simhei.ttf"')


# In[3]:


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[13]:


data2 = pd.read_excel(r'C:\Users\Administrator\Desktop/新城市活力归一化.xlsx', sheet_name = '实空间相关性')
data2.set_index('指标',drop = True,inplace = True)
data2.info()


# In[48]:


corr = filled_df.corr()
fig, ax = plt.subplots()
fig.set_size_inches(30, 20)
#sns.set(font=zhfont.get_name())

plt.tick_params(axis='both',which='major',labelsize=16)

fig = sns.heatmap(corr, vmin=-1, vmax=1,annot=True, center=0, linewidths=0.7)
fig.set_title("指标相关性分析")
heatmap = fig.get_figure()


# # 聚类

# ## 总分类

# In[6]:


# 3 clusters
df_norm = filled_df.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(df_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=df_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[7]:


kmeanscluster = KMeans(n_clusters= 4, init = 'k-means++')
kmeanscluster.fit(df_norm)
Y_pred = kmeanscluster.predict(df_norm)

""" now trying Dendrogram """
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(df_norm, method= 'ward', metric = 'euclidean'))
# Selecting 3


# In[8]:


from scipy.cluster.hierarchy import linkage,dendrogram  # scipy中的层次聚类

Z=linkage(filled_df,method='ward',metric='euclidean')
#method={ ‘single’,‘complete’, ‘average’, ‘weighted’, ‘centroid’, ‘median’,‘ward’ }
#metric={ ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’}
P=dendrogram(Z,0) #系统树图
plt.show()  # 画出聚类图


# In[9]:


import plotly.figure_factory as ff
names = df_norm.index
fig = ff.create_dendrogram(filled_df, orientation='left', labels=filled_df.index)
fig.update_layout(width=800, height=800)
fig.show()
#fig.savefig('总.jpeg',dpi = 800)


# # 自然

# In[10]:


nature = filled_df[filled_df.columns[0:4]]
nature.head()


# In[11]:


nature_norm = nature.apply(preprocessing.scale,axis=0)
names = nature_norm.index
fig = ff.create_dendrogram(nature_norm, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[26]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns=nature_norm.columns)
pd.set_option('precision', 3)
print(centroids)
pd.set_option('precision', 6)


# In[29]:


withinClusterSS = [0] * 4
clusterCount = [0] * 4
for cluster, distance in zip(kmeans.labels_, kmeans.transform(nature_norm)):
    withinClusterSS[cluster] += distance[cluster]**2
    clusterCount[cluster] += 1
for cluster, withClustSS in enumerate(withinClusterSS):
    print('Cluster {} ({} members): {:5.2f} within cluster'.format(cluster, 
        clusterCount[cluster], withinClusterSS[cluster]))


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/liberation/simhei.ttf')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# In[31]:


centroids['cluster'] = ['Cluster {}'.format(i) for i in centroids.index]
import matplotlib
matplotlib.rc("font",family='YouYuan')

ax = parallel_coordinates(centroids, class_column='cluster', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))

centroids


# # 人

# In[12]:


pop = filled_df[filled_df.columns[5:10]]
pop.head()


# In[13]:


pop_norm = pop.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(pop_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=pop_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[14]:



import plotly.figure_factory as ff
names = pop.index
fig = ff.create_dendrogram(pop, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# # 企业

# In[15]:


company = filled_df[filled_df.columns[10:16]]

company_norm = company.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(company_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=company_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[16]:


company.head()


# In[17]:


names = company_norm.index
fig = ff.create_dendrogram(company_norm, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# # "实"空间

# In[18]:


real = filled_df[filled_df.columns[16:22]]

real_norm = real.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=3, random_state=0).fit(real_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=real_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[19]:


real.head()


# In[20]:


names = real_norm.index
fig = ff.create_dendrogram(real_norm, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# # “虚”空间

# In[21]:


virtual = filled_df[filled_df.columns[22:26]]

virtual_norm = virtual.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=3, random_state=0).fit(virtual_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=virtual_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[22]:


virtual.head()


# In[23]:


names = virtual_norm.index
fig = ff.create_dendrogram(virtual_norm, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[24]:


virtual = filled_df[filled_df.columns[26:30]]

virtual_norm = virtual.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(virtual_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=virtual_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[25]:


virtual


# In[26]:


names = virtual_norm.index
fig = ff.create_dendrogram(virtual_norm, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# # 创造力分类

# In[27]:


creativity = filled_df[['万人拥有企业专利申请量','万人知网论文阅读量','万人拥有微信公众号量']]
creativity.head()


# In[28]:


import plotly.figure_factory as ff
names = creativity.index
fig = ff.create_dendrogram(creativity, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[12]:


filled_df.columns


# # 成长力

# In[29]:


grow = filled_df[['万人拥有小微企业数量','万人拥有高新技术企业数量','上市公司数量','营商环境指数']]
names = grow.index
fig = ff.create_dendrogram(grow, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# # 活跃度

# In[32]:


activity =  filled_df[['夜间活力占比','微博活跃度','百度搜索指数','文化热度指数']]
names = grow.index
fig = ff.create_dendrogram(activity, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[33]:



kmeans = KMeans(n_clusters=4, random_state=0).fit(activity)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=activity.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# # 设施

# In[34]:


fac =  filled_df[['万人拥有体育设施数量','万人拥有文化旅游设施数量','万人拥有商业服务设施数量']]
names = fac.index
fig = ff.create_dendrogram(fac, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[18]:


kmeans = KMeans(n_clusters=4, random_state=0).fit(fac)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=fac.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# # 中原特色

# In[35]:


cul =  filled_df[['蜜雪冰城指数','烩面指数','胡辣汤指数']]
names = cul.index
fig = ff.create_dendrogram(cul, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[20]:


kmeans = KMeans(n_clusters=4, random_state=0).fit(cul)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=cul.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# # 设施功能性

# In[36]:


real = filled_df[filled_df.columns[18:22]]

real_norm = real.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(real_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=real_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[37]:


real.head()


# In[38]:


names = real.index
fig = ff.create_dendrogram(real, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# # 宜居性

# In[39]:


cul =  filled_df[['全年晴天天数','舒适度指数','空气质量优良天数','高铁联系指数','通勤时间指数'
                 ]]
names = cul.index
fig = ff.create_dendrogram(cul, orientation='left', labels=names)
fig.update_layout(width=800, height=800)
fig.show()


# In[48]:


cul_norm = cul.apply(preprocessing.scale,axis=0)
kmeans = KMeans(n_clusters=4, random_state=0).fit(cul_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=cul_norm.index)
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# # 归一化

# In[58]:


normalized_df= 1 + 9/(filled_df.max()-filled_df.min())*(filled_df-filled_df.min())
normalized_df.to_excel("新城市活力归一化.xlsx")


# # 相关性分析

# In[8]:


import matplotlib.pyplot as plt
import matplotlib as mpl
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/liberation/simhei.ttf')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号corr = updated_df6.corr()


# In[45]:


corr = pop.corr()
fig, ax = plt.subplots()
fig.set_size_inches(30, 20)

pop.reset_index(drop = True)
fig = sns.heatmap(corr, vmin=0, vmax=1,cmap = 'Spectral',annot=True, center=0, linewidths=0.7)
fig.set_title("人口指标相关性分析",fontsize = 35)

#heatmap.savefig('corr', dpi = 800)
#sns.heatmap(corr, annot=True, center=0,ax=ax)
plt.tick_params(axis='both',which='major',labelsize=22)
plt.legend(frameon=False, loc='best',fontsize = 20)
plt.tick_params(axis='both',which='major',labelsize=22)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[42]:


plt.savefig('corr.png', dpi = 400)


# In[44]:


pop = filled_df[filled_df.columns[4:10]]
pop.reset_index(drop=True, inplace=True)
pop.head()


# In[76]:


data1 = pd.read_excel(r'C:\Users\Administrator\Desktop/自然和活力.xlsx',sheet_name = 'Sheet2')
filled_df = data1.copy()
#filled_df.loc[:, '接待入境游客人数（人次）'].replace([0], [41519], inplace=True)
#filled_df['全社会R&D支出占GDP比重'] = filled_df['全社会R&D支出占GDP比重'].str.strip('%').astype(float) / 100
#filled_df = filled_df.drop(['编号'],axis = 1)
#filled_df.set_index(['地市名称'], inplace=True)
filled_df.head(10)


# In[61]:


filled_df.reset_index(drop=True, inplace=True)


# In[77]:


corr = filled_df.corr()
fig, ax = plt.subplots()
fig.set_size_inches(30, 20)

fig = sns.heatmap(corr, vmin=-1, vmax=1,cmap = 'Spectral',annot=True, center=0, linewidths=0.7)
#fig.set_title("人口指标相关性分析",fontsize = 35)

#heatmap.savefig('corr', dpi = 800)
#sns.heatmap(corr, annot=True, center=0,ax=ax)
plt.tick_params(axis='both',which='major',labelsize=22)
plt.legend(frameon=False, loc='best',fontsize = 20)
plt.tick_params(axis='both',which='major',labelsize=22)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[ ]:




