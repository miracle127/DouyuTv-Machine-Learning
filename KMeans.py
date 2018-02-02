import pandas as pd
from sklearn.cluster import KMeans
data = pd.read_csv('/home/miracle/文档/机器学习数据.csv')
d = data.ix[:,2:]
#data_zs = 1.0*(d - d.median())/d.std()
d0 = d[data['时段']==0]
d0_zs = 1.0*(d0-d0.mean())/d0.std()
d1 = d[data['时段']==1]
d1_zs = 1.0*(d1-d1.mean())/d1.std()
d2 = d[data['时段']==2]
d2_zs = 1.0*(d2-d2.mean())/d2.std()

k = 5
iteration = 1000
model = KMeans(n_clusters=k,max_iter=iteration,n_jobs=4)
model.fit(d0_zs)
tmp1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
tmp2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
tmp2.columns = d0.columns
tmp2 = tmp2*d0.std()+d0.mean()
r0 = pd.concat([tmp2, tmp1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
d0['类别'] = model.labels_
d0['时段'] = '午夜档'

model = KMeans(n_clusters=k,max_iter=iteration,n_jobs=4)
model.fit(d1_zs)
tmp1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
tmp2 = pd.DataFrame(model.cluster_centers_) #
tmp2.columns = d1.columns
tmp2 = tmp2*d1.std()+d1.mean()
r1 = pd.concat([tmp2, tmp1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
d1['类别'] = model.labels_
d1['时段'] = '白天档'

model = KMeans(n_clusters=k,max_iter=iteration,n_jobs=4)
model.fit(d2_zs)
tmp1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
tmp2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
tmp2.columns = d2.columns
tmp2 = tmp2*d2.std()+d2.mean()
r2 = pd.concat([tmp2, tmp1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
d2['类别'] = model.labels_
d2['时段'] = '晚间档'

r = pd.concat([r0,r1,r2])
r.columns = list(d0.columns[:-2]) + [u'类别数目'] #重命名表头
d = pd.concat([d0,d1,d2])
r.to_csv('/home/miracle/文档/KMeans聚类中心统计.csv')
d.to_csv('/home/miracle/文档/斗鱼机器学习(带类标记).csv')
