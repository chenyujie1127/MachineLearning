"""
LR: 逻辑斯谛回归

"""

from math import exp
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 构建鸢尾花数据集
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns = iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    return data[:,:2],data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# LR模型
class LogisticRegressionClassifier:
    def __init__(self, max_iter = 200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
    
    def sigmoid(self,x):
        return 1 / (1 + exp(-x))
    
    def data_matrix(self,X):
        data_mat = []
        for d in x:
            data_mat.append([1.0,*d])
        return data_mat
    
    def fit(self,X,y):
        data_mat = self.data_matrix(X)
        self.weights = np.zeros((len(data_mat[0]),1),dtype=np.float32)
        # 迭代多少次
        for iter_ in range(self.max_iter):
            # 所有数据集
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i],self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
        
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            self.learning_rate, self.max_iter))

    def score(self,X_test,y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x,y in zip(X_test,y_test):
            result = np.dot(x,self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)

lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)
# LogisticRegression Model(learning_rate=0.01,max_iter=200)
lr_clf.score(X_test, y_test)

######## scikit-learn ###########

"""

sklearn.linear_model.LogisticRegression

1. solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：

a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。SAG是一种线性收敛算法，这个速度远比SGD快

2. penalty参数的选择会影响我们损失函数优化算法的选择。

即参数solver的选择，如果是L2正则化，那么4种可选的算法{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}都可以选择。
但是如果penalty是L1正则化的话，就只能选择‘liblinear’了。
这是因为L1正则化的损失函数不是连续可导的，而{‘newton-cg’, ‘lbfgs’,‘sag’}这三种优化算法时都需要损失函数的一阶或者二阶连续导数。
而‘liblinear’并没有这个依赖。

选择：
small dataset or l1 penalty: "liblinear"
Multinomial loss or large dataset: {‘newton-cg’, ‘lbfgs’,‘sag’}
very large dataset: sag

"""
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.coef_, clf.intercept_)
# [[ 2.59546005 -2.81261232]] [-5.08164524]

