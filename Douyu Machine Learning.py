import pandas as pd
from sklearn.model_selection import cross_val_score,KFold,StratifiedShuffleSplit
import numpy as np
seed = 42
np.random.seed(seed)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
score = pd.DataFrame()
import pydotplus
from sklearn.tree import export_graphviz
data = pd.read_csv('/home/miracle/文档/斗鱼机器学习(带类标记).csv',index_col=0)
data['时段'][data['时段']=='午夜档']=0
data['时段'][data['时段']=='白天档']=1
data['时段'][data['时段']=='晚间档']=2
y = data['类别'].as_matrix()
del(data['类别'])
x = data.as_matrix()
ss = StratifiedShuffleSplit(n_splits=1,test_size=0.25,train_size=0.75,random_state=0)
for train_index,test_index in ss.split(x,y):
    x_train,x_test = x[train_index],x[test_index]
    y_train,y_test = y[train_index],y[test_index]
    print(x_train,x_test,y_train,y_test)
#运用不同学习算法训练不同的分类器,并用ROC曲线与混淆矩阵评价分类器性能
#0、决策树
print('决策树')
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
scores = cross_val_score(tree,x_test,y_test,cv=kfold)
score = score.append([[scores.mean()]])
dot_data = export_graphviz(tree,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('tree.pdf')
#1、随机森林
print('随机森林')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=80,max_features=0.6)
rfc.fit(x_train,y_train)
scores = cross_val_score(rfc,x_test,y_test,cv=kfold)
score = score.append([[scores.mean()]])
#2、极端随机树
print('极端随机树')
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=80,max_features=0.6)
etc.fit(x_train,y_train)
scores = cross_val_score(etc,x_test,y_test,cv=kfold)
score = score.append([[scores.mean()]])
#3、支持向量机
print('支持向量机')
from sklearn import svm
svm = svm.SVC(decision_function_shape='ovr')
svm.fit(x_train,y_train)
scores = cross_val_score(svm,x_test,y_test,cv=kfold)
score = score.append([[scores.mean()]])

#4、神经网络
print('神经网络')
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
def base_model():
    model = Sequential()
    model.add(Dense(units=14,input_dim=8,activation='relu'))
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
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
estimator = KerasClassifier(build_fn=base_model,epochs=50,batch_size=256)
estimator.fit(x_train,y_train,epochs=100,batch_size=256)
scores0 = cross_val_score(estimator,x_test,y_test,cv=kfold)
score = score.append([[scores.mean()]])

print(score)
from cm_plot import *
cm_plot(y_test,tree.predict(x_test)).show()
cm_plot(y_test,rfc.predict(x_test)).show()
cm_plot(y_test,etc.predict(x_test)).show()
cm_plot(y_test,svm.predict(x_test)).show()
cm_plot(y_test,estimator.predict(x_test)).show()