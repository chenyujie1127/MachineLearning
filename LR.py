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


"""
LR

"""
import numpy as np
import time

class LogisticRegression:
    def __init__(self,learn_rate=0.1,max_iter=10000,tol=1e-2):
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        # 迭代停止阈值
        self.tol = tol
        # 权重
        self.w = None

    def preprocessing(self,X):
        """将原始X末尾加上一列，该列数值全部为1,bias?"""
        row = X.shape[0]
        y = np.ones(row).reshape(row,1)
        X_prepro = np.hstack((X,y))
        return X_prepro

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))

    def fit(self,X_train,y_train):
        X = self.preprocessing(X_train)
        y = y_train.T
        # 初始化权重w
        self.w = np.array([[0] * X.shape[1]], dtype=np.float)

        k = 0
        for loop in range(self.max_iter):
            # 计算梯度
            z = np.dot(X, self.w.T)
            grad = X * (y-self.sigmoid(z))
            grad = grad.sum(axis = 0)
            # 利用梯度的绝对值作为迭代中止的条件
            if (np.abs(grad) <= self.tol).all():
                break
            else:
                # 更新权重w 梯度上升——求极大值
                self.w += self.learn_rate * grad
                k += 1
        print("迭代次数：{}次".format(k))
        print("最终梯度：{}".format(grad))
        print("最终权重：{}".format(self.w[0]))

    def predict(self,x):
        p = self.sigmoid(np.dot(self.preprocessing(x),self.w.T))
        # print("Y=1的概率被估计为：{:.2%}".format(p[0][0]))  # 调用score时，注释掉
        p[np.where(p>0.5)] = 1
        p[np.where(p<0.5)] = 0
        return p
    
    def score(self,X,y):
        y_c = self.predict(X)
        error_rate = np.sum(np.abs(y_c-y.T)) / y_c.shape[0]
        return 1 - error_rate

# 训练数据集
X_train = np.array([[3, 3, 3], [4, 3, 2], [2, 1, 2], [1, 1, 1], [-1, 0, 1],
                    [2, -2, 1]])
y_train = np.array([[1, 1, 1, 0, 0, 0]])
# 构建实例，进行训练
clf = LogisticRegression()
clf.fit(X_train, y_train)

"""
迭代次数：3232次
最终梯度：[ 0.00144779  0.00046133  0.00490279 -0.00999848]
最终权重：[  2.96908597   1.60115396   5.04477438 -13.43744079]

"""
