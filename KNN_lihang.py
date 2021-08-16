"""
K 近邻 算法

k近邻法是基本且简单的分类与回归方法。$k$近邻法的基本做法是：对给定的训练实例点和输入实例点，首先确定输入实例点的k个最近邻训练实例点，然后利用这k个训练实例点的类的多数来预测输入实例点的类

K近邻模型对应于基于训练数据集对特征空间的一个划分

近邻法三要素：距离度量、K值的选择和分类决策规则。

（1）常用的距离度量是欧氏距离及更一般的pL距离。 【曼哈顿距离: p=1; 欧式距离: p=2; 切比雪夫距离: p=无穷大】

（2）K值小时，K近邻模型更复杂；K值大时，K近邻模型更简单。K值的选择反映了对近似误差与估计误差之间的权衡，通常由交叉验证选择最优的K。

（3）常用的分类决策规则是多数表决，对应于经验风险最小化。

K近邻法的实现需要考虑如何快速搜索k个最近邻点。
-- kd树是一种便于对k维空间中的数据进行快速检索的数据结构。
-- kd树是二叉树，表示对K维空间的一个划分，其每个结点对应于K维空间划分中的一个超矩形区域。
利用kd树可以省去对大部分数据点的搜索， 从而减少搜索的计算量。

"""

###### 距离度量 ######
import math
from itertools import combinations

def L(x,y,p=2):
    if len(x) == len(y) and len(x)>1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i]-y[i]),p)
        return math.pow(sum,1/p)
    else:
        return 0

###### 加载数据 #######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from collections import Counter

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df)

# df.iloc: 提取行列，返回DataFrame类型
data = np.array(df.iloc[:100,[0,1,-1]])
X,y = data[:,:-1],data[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


###### 定义KNN #######

class KNN:
    
    def __init__(self,X_train,y_train,n_neighbors = 3, p = 2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self,X):
        # 取n个点
        knn_list = []
        
        # 先存上n个
        for i in range(self.n):
            # linalg = linear（线性）+algebra（代数），norm则表示范数
            # np.linalg.norm(x, ord=None, axis=None, keepdims=False)
            # x: 表示矩阵（也可以是一维）; ord: 范数类型; axis: 处理类型
            dist = np.linalg.norm(X-self.X_train[i],ord=self.p)
            knn_list.append((dist,self.y_train[i]))
        
        # 对于剩下的点,逐一替换
        for i in range(self.n,len(X_train)):
            # 找到距离最远的替换
            max_index = knn_list.index(max(knn_list,key=lambda x:x[0]))
            dist = np.linalg.norm(X-self.X_train[i],ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist,self.y_train[i])
        
        # 统计这n个点中的分类,选择分类最多的返回
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(),key = lambda x:x[1])[-1][0]
        return max_count

    # 测试: 对每一个结果进行遍历
    def score(self,X_test,y_test):
        right_count = 0
        n = 10
        for X,y in zip(X_test,y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train,y_train)
clf.score(X_test,y_test)

test_point = [6.0,3.0]
print('Test Point: {}'.format(clf.predict(test_point)))

###### 使用scikit-learn实现 #######

"""
sklearn.neighbors.KNeighborsClassifier
参数:
n_neighbors: 临近点个数; 
p: 距离度量; 
algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}; 
weights: 确定近邻的权重
"""

from sklearn.neighbors import KNeighborsClassifier
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train,y_train)
clf_sk.score(X_test,y_test)


###### 构造平衡Kd树算法 #######

"""
Kd树: 

kd树是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。

kd树是二叉树，表示对K维空间的一个划分（partition）。构造kd树相当于不断地用垂直于坐标轴的超平面将K维空间切分，构成一系列的k维超矩形区域。kd树的每个结点对应于一个K维超矩形区域。

构造kd树的方法如下：

构造根结点，使根结点对应于K维空间中包含所有实例点的超矩形区域；通过下面的递归方法，不断地对K维空间进行切分，生成子结点。

在超矩形区域（结点）上选择一个坐标轴和在此坐标轴上的一个切分点，确定一个超平面，这个超平面通过选定的切分点并垂直于选定的坐标轴，将当前超矩形区域切分为左右两个子区域 （子结点）；

这时，实例被分到两个子区域。这个过程直到子区域内没有实例时终止（终止时的结点为叶结点）。在此过程中，将实例保存在相应的结点上。

通常，依次选择坐标轴对空间切分，选择训练实例点在选定坐标轴上的中位数 （median）为切分点，这样得到的kd树是平衡的。注意，平衡的kd树搜索时的效率未必是最优的。

"""

class KdNode(obejct):
    def __init__(self,dom_elt,split,left,right):
        # dom_elt: k维向量节点(k维空间中的一个样本点)
        # split: 整数（进行分割维度的序号）
        # left : 该结点分割超平面左子空间构成的kd-tree
        # right: 该结点分割超平面右子空间构成的kd-tree
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right

class KdTree(obejct):

    def __init__(self,data):
        # 存储数据总共多少维
        k = len(data[0])
        # 按第split维划分数据集exset创建KdNode
        def CreateNode(split,data_set):
            if not data_set:
                return None
            # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            # 找中位切分点
            split_pos = len(data_set) // 2
            # 找到中间那个节点
            median = data_set[split_pos]
            split_next = (split + 1) % k

            return KdNode(
                median,
                split,
                CreateNode(split_next,data_set[:split_pos]) # 创建左子树
                CreateNode(split_next,data_set[split_pos+1:]) # 创建右子树
            )
        # 从第0维分量开始构建kd树,返回根节点
        self.root = CreateNode(0,data)


# KDTree的前序遍历
def preOrder(root):
    print(root.dom_elt)
    if root.left:
        preOrder(root.left)
    if root.right:
        preOrder(root.right)


###### 对构建好的kd树进行搜索，寻找与目标点最近的样本点 #######

"""
如: 找x的最近邻

在kd树中找出包含目标点x的叶结点：

从根结点出发，递归地向下访问树。

若目标点x当前维的坐标小于切分点的坐标，则移动到左子结点，否则移动到右子结点，直到子结点为叶结点为止；

如果“当前k近邻点集”元素数量小于k或者叶节点距离小于“当前k近邻点集”中最远点距离，那么将叶节点插入“当前k近邻点集”；

递归地向上回退，在每个结点进行以下操作：

(a)如果“当前k近邻点集”元素数量小于k或者当前节点距离小于“当前k近邻点集”中最远点距离，那么将该节点插入“当前k近邻点集”。

(b)检查另一子结点对应的区域是否与以目标点为球心、以目标点与于“当前k近邻点集”中最远点间的距离为半径的超球体相交。

如果相交，可能在另一个子结点对应的区域内存在距目标点更近的点，移动到另一个子结点，接着，递归地进行最近邻搜索；如果不相交，向上回退；

当回退到根结点时，搜索结束，最后的“当前k近邻点集”即为x的最近邻点。

"""

from math import sqrt
from collections import namedtuple


# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
# collections.namedtuple(typename, field_names, verbose=False, rename=False) 
# typename：元组名称; field_names: 元组中元素的名称; rename: 如果元素名称中含有 python 的关键字，则必须设置为 rename=True; verbose: 默认就好
result = namedtuple("Result_tuple",
                    "nearest_point  nearest_dist  nodes_visited")

def find_nearest(tree,point):
    # 数据维度
    k = len(point)

    def travel(kd_node,target,max_dist):
        if kd_node is None:
            return result([0]*k,float("inf"),0)
        nodes_visited = 1
        # 进行分割的维度
        s = kd_node.split
        # 进行分割的“轴” 【即某一数据】
        pivot = kd_node.dom_elt

        # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
        if target[s] <= pivot[s]:
            nearer_node = kd_node.left
            furthur_node = kd_node.right
        else:
            nearer_node = kd.right
            furthur_node = kd.left

        # 遍历找到包含目标点的区域
        temp1 = travel(nearer_node,target,max_dist)
        
        # 以此叶结点作为“当前最近点”
        nearest = temp1.nearest_point
        # 更新最近距离
        dist = temp1.nearest_dist
        nodes_visited += temp1.nodes_visited
        # 最近点将在以目标点为球心，max_dist为半径的超球体内
        if dist < max_dist:
            max_dist = dist
        # 第s维上目标点与分割超平面的距离
        temp_dist = abs(pivot[s]-target[s])
         # 判断超球体是否与超平面相交
        if max_dist < temp_dist:
            # 不相交则可以直接返回，不用继续判断
            return result(nearest,dist,nodes_visited)

        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1-p2)**2 for p1,p2 in zip(pivot,target)))

        # 如果“更近”, 更新最近点，更新最近距离，更新超球体半径
        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist
        
        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(furthur_node,target,max_dist)
        nodes_visited += temp2.nodes_visited

        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist
        
        return result(nearest,dist,nodes_visited)

    # 从根节点开始递归
    return travel(tree.root,point,float("inf"))
        
data = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
kd = KdTree(data)
preorder(kd.root)

from time import clock
from random import random

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]
 
# 产生n个k维随机向量 
def random_points(k, n):
    return [random_point(k) for _ in range(n)]

ret = find_nearest(kd, [3,4.5])
print (ret)
# Result_tuple(nearest_point=[2, 3], nearest_dist=1.8027756377319946, nodes_visited=4)

N = 400000
t0 = clock()
kd2 = KdTree(random_points(3, N))            # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest(kd2, [0.1,0.5,0.8])      # 四十万个样本点中寻找离目标最近的点
t1 = clock()
print ("time: ",t1-t0, "s")
print (ret2)
# time:  5.170202400000001 s
# Result_tuple(nearest_point=[0.09916902877403755, 0.5005978535517558, 0.7997848590100571], nearest_dist=0.0010460533893058112, nodes_visited=38)

###### 使用sklearn.neighbors ###### 
import numpy as np
from sklearn.neighbors import KDTree

train_data = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
tree = KDTree(train_data, leaf_size=2)
dist, ind = tree.query(np.array([(3, 4.5)]), k=1)
x1 = train_data[ind[0]][0][0]
x2 = train_data[ind[0]][0][1]

print("x点的最近邻点是({0}, {1})".format(x1, x2))

