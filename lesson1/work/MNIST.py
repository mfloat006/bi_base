# 1.导包
import numpy as np
import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# 2.数据加载
digits = load_digits()
data = digits.data

# 3.数据预处理
# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)
# Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

clf = DecisionTreeClassifier()
clf.fit(train_x, train_y)

# 4.模型评估
y_train_predict = clf.predict(train_x)
y_test_predict = clf.predict(test_x)
print('训练集准确率：%0.4lf'%accuracy_score(train_y,y_train_predict))
print('训练集准确率：%0.4lf'%accuracy_score(test_y,y_test_predict))

# 5.模型优化
for i in range(3,15):
	clf = DecisionTreeClassifier(

		max_depth=i,
		)
	clf.fit(train_x, train_y)

	# 5.优化
	y_train_predict = clf.predict(train_x)
	y_test_predict = clf.predict(test_x)
	print('max_depth:', i)
	print('训练集准确率：%0.4lf'%accuracy_score(train_y,y_train_predict))
	print('训练集准确率：%0.4lf'%accuracy_score(test_y,y_test_predict))