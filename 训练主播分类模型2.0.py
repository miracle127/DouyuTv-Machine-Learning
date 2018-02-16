import pandas as pd
from sklearn.model_selection import cross_val_score,KFold,StratifiedShuffleSplit
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.naive_bayes import GaussianNB as GNB
from cm_plot import *

seed = 42
np.random.seed(seed)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
score = pd.DataFrame()
#import pydotplus
#from sklearn.tree import export_graphviz
data = pd.read_csv('/home/miracle/文档/斗鱼机器学习(带类标记).csv',index_col=0)
data['时段'][data['时段']=='午夜档']=0
data['时段'][data['时段']=='白天档']=1
data['时段'][data['时段']=='晚间档']=2
d0 = data[data['时段']==0].iloc[:,:-1]
d1 = data[data['时段']==1].iloc[:,:-1]
d2 = data[data['时段']==2].iloc[:,:-1]
d0.iloc[:,:-1] = (d0.iloc[:,:-1]-d0.iloc[:,:-1].mean())/d0.iloc[:,:-1].std()
d1.iloc[:,:-1] = (d1.iloc[:,:-1]-d1.iloc[:,:-1].mean())/d1.iloc[:,:-1].std()
d2.iloc[:,:-1] = (d2.iloc[:,:-1]-d2.iloc[:,:-1].mean())/d2.iloc[:,:-1].std()
for i in (d0['类别'].value_counts()[d0['类别'].value_counts()==1].index):
    d0 = d0.append(d0[d0['类别']==i],ignore_index=True)
for i in (d1['类别'].value_counts()[d1['类别'].value_counts()==1].index):
    d1 = d1.append(d1[d1['类别']==i],ignore_index=True)
for i in (d2['类别'].value_counts()[d2['类别'].value_counts()==1].index):
    d2 = d2.append(d2[d2['类别']==i],ignore_index=True)
d0_matrix = d0.as_matrix()
d1_matrix = d1.as_matrix()
d2_matrix = d2.as_matrix()
ss = StratifiedShuffleSplit(n_splits=1,test_size=0.5,train_size=0.5,random_state=0)
for train_index,test_index in ss.split(d0.iloc[:,:-1].as_matrix(),d0.iloc[:,-1].as_matrix()):
    d0train_x,d0test_x = d0.iloc[:,:-1].as_matrix()[train_index],d0.iloc[:,:-1].as_matrix()[test_index]
    d0train_y,d0test_y = d0.iloc[:,-1].as_matrix()[train_index],d0.iloc[:,-1].as_matrix()[test_index]
for train_index,test_index in ss.split(d1.iloc[:,:-1].as_matrix(),d1.iloc[:,-1].as_matrix()):
    d1train_x,d1test_x = d1.iloc[:,:-1].as_matrix()[train_index],d1.iloc[:,:-1].as_matrix()[test_index]
    d1train_y,d1test_y = d1.iloc[:,-1].as_matrix()[train_index],d1.iloc[:,-1].as_matrix()[test_index]
for train_index,test_index in ss.split(d2.iloc[:,:-1].as_matrix(),d2.iloc[:,-1].as_matrix()):
    d2train_x,d2test_x = d2.iloc[:,:-1].as_matrix()[train_index],d2.iloc[:,:-1].as_matrix()[test_index]
    d2train_y,d2test_y = d2.iloc[:,-1].as_matrix()[train_index],d2.iloc[:,-1].as_matrix()[test_index]
#运用不同学习算法训练不同的分类器,并用ROC曲线与混淆矩阵评价分类器性能
#0、决策树
print('决策树')
tree0 = DecisionTreeClassifier()
tree1 = DecisionTreeClassifier()
tree2 = DecisionTreeClassifier()
tree0.fit(d0train_x,d0train_y)
tree1.fit(d1train_x,d1train_y)
tree2.fit(d2train_x,d2train_y)
scores0 = cross_val_score(tree0,d0test_x,d0test_y,cv=kfold)
scores1 = cross_val_score(tree1,d1test_x,d1test_y,cv=kfold)
scores2 = cross_val_score(tree2,d2test_x,d2test_y,cv=kfold)
score = score.append([[scores0.mean(),scores1.mean(),scores2.mean()]])

cm_plot(d0test_y,tree0.predict(d0test_x)).savefig('决策树0.png')
cm_plot(d1test_y,tree1.predict(d1test_x)).savefig('决策树1.png')
cm_plot(d2test_y,tree2.predict(d2test_x)).savefig('决策树2.png')
#dot_data = export_graphviz(tree,out_file=None)
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf('tree.pdf')
#1、随机森林
print('随机森林')
rfc0 = RandomForestClassifier(n_estimators=80,max_features=0.6)
rfc1 = RandomForestClassifier(n_estimators=80,max_features=0.6)
rfc2 = RandomForestClassifier(n_estimators=80,max_features=0.6)
rfc0.fit(d0train_x,d0train_y)
rfc1.fit(d1train_x,d1train_y)
rfc2.fit(d2train_x,d2train_y)
scores0 = cross_val_score(rfc0,d0test_x,d0test_y,cv=kfold)
scores1 = cross_val_score(rfc1,d1test_x,d1test_y,cv=kfold)
scores2 = cross_val_score(rfc2,d2test_x,d2test_y,cv=kfold)
score = score.append([[scores0.mean(),scores1.mean(),scores2.mean()]])
cm_plot(d0test_y,rfc0.predict(d0test_x)).savefig('随机森林0.png')
cm_plot(d1test_y,rfc1.predict(d1test_x)).savefig('随机森林1.png')
cm_plot(d2test_y,rfc2.predict(d2test_x)).savefig('随机森林2.png')
#2、极端随机树
print('极端随机树')
etc0 = ExtraTreesClassifier(n_estimators=80,max_features=0.6)
etc1 = ExtraTreesClassifier(n_estimators=80,max_features=0.6)
etc2 = ExtraTreesClassifier(n_estimators=80,max_features=0.6)
etc0.fit(d0train_x,d0train_y)
etc1.fit(d1train_x,d1train_y)
etc2.fit(d2train_x,d2train_y)
scores0 = cross_val_score(etc0,d0test_x,d0test_y,cv=kfold)
scores1 = cross_val_score(etc1,d1test_x,d1test_y,cv=kfold)
scores2 = cross_val_score(etc2,d2test_x,d2test_y,cv=kfold)
score = score.append([[scores0.mean(),scores1.mean(),scores2.mean()]])
cm_plot(d0test_y,etc0.predict(d0test_x)).savefig('极端随机树0.png')
cm_plot(d1test_y,etc1.predict(d1test_x)).savefig('极端随机树1.png')
cm_plot(d2test_y,etc2.predict(d2test_x)).savefig('极端随机树2.png')
#3、支持向量机
print('支持向量机')
svm0 = svm.SVC(decision_function_shape='ovr')
svm1 = svm.SVC(decision_function_shape='ovr')
svm2 = svm.SVC(decision_function_shape='ovr')
svm0.fit(d0train_x,d0train_y)
svm1.fit(d1train_x,d1train_y)
svm2.fit(d2train_x,d2train_y)
scores0 = cross_val_score(svm0,d0test_x,d0test_y,cv=kfold)
scores1 = cross_val_score(svm1,d1test_x,d1test_y,cv=kfold)
scores2 = cross_val_score(svm2,d2test_x,d2test_y,cv=kfold)
score = score.append([[scores0.mean(),scores1.mean(),scores2.mean()]])
cm_plot(d0test_y,svm0.predict(d0test_x)).savefig('支持向量机0.png')
cm_plot(d1test_y,svm1.predict(d1test_x)).savefig('支持向量机1.png')
cm_plot(d2test_y,svm2.predict(d2test_x)).savefig('支持向量机2.png')

#4、高斯朴素贝叶斯
print('高斯朴素贝叶斯')
gnb0 = GNB()
gnb1 = GNB()
gnb2 = GNB()
gnb0.fit(d0train_x,d0train_y)
gnb1.fit(d1train_x,d1train_y)
gnb2.fit(d2train_x,d2train_y)
scores0 = cross_val_score(gnb0,d0test_x,d0test_y,cv=kfold)
scores1 = cross_val_score(gnb1,d1test_x,d1test_y,cv=kfold)
scores2 = cross_val_score(gnb2,d2test_x,d2test_y,cv=kfold)
score = score.append([[scores0.mean(),scores1.mean(),scores2.mean()]])
cm_plot(d0test_y,gnb0.predict(d0test_x)).savefig('高斯朴素贝叶斯0.png')
cm_plot(d1test_y,gnb1.predict(d1test_x)).savefig('高斯朴素贝叶斯1.png')
cm_plot(d2test_y,gnb2.predict(d2test_x)).savefig('高斯朴素贝叶斯2.png')

#5、神经网络
print('神经网络')
y0train = np_utils.to_categorical(d0train_y)
y0test = np_utils.to_categorical(d0test_y)
y1train = np_utils.to_categorical(d1train_y)
y1test = np_utils.to_categorical(d1test_y)
y2train = np_utils.to_categorical(d2train_y)
y2test = np_utils.to_categorical(d2test_y)
def base_model():
    model = Sequential()
    model.add(Dense(units=14,input_dim=7,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=14,input_dim=14,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=14,input_dim=14,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=14,input_dim=14,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=5,input_dim=14,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
estimator0 = KerasClassifier(build_fn=base_model,epochs=50,batch_size=256)
estimator0.fit(d0train_x,y0train,epochs=100,batch_size=256)
estimator1 = KerasClassifier(build_fn=base_model,epochs=50,batch_size=256)
estimator1.fit(d1train_x,y1train,epochs=100,batch_size=256)
estimator2 = KerasClassifier(build_fn=base_model,epochs=50,batch_size=256)
estimator2.fit(d2train_x,y2train,epochs=100,batch_size=256)
scores0 = cross_val_score(estimator0,d0test_x,y0test,cv=kfold)
scores1 = cross_val_score(estimator1,d1test_x,y1test,cv=kfold)
scores2 = cross_val_score(estimator2,d2test_x,y2test,cv=kfold)
score = score.append([[scores0.mean(),scores1.mean(),scores2.mean()]])

cm_plot(d0test_y,estimator0.predict(d0test_x)).savefig('神经网络0.png')
cm_plot(d1test_y,estimator1.predict(d1test_x)).savefig('神经网络1.png')
cm_plot(d2test_y,estimator2.predict(d2test_x)).savefig('神经网络2.png')

score.index = ['决策树','随机森林','极端随机树','支持向量机','高斯朴素贝叶斯','神经网络']
score.columns = ['午夜档','白天档','夜间档']
print(score)
score.to_csv('/home/miracle/文档/斗鱼机器学习模型交叉检验评分.csv')