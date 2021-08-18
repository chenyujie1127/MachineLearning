"""
聚类方法：

聚类是针对给定的样本，依据它们属性的相似度或距离，将其归并到若干个“类”或“簇”的数据分析问题。一个类是样本的一个子集。直观上，相似的样本聚集在同类，不相似的样本分散在不同类。

常用的距离度量：曼哈顿距离、欧式距离、切比雪夫距离
常用的相似度量：相关系数、夹角余弦
用距离度量相似度时，距离越小表示样本越相似；用相关系数时，相关系数越大表示样本越相似。

描述某类的特征的指标有中心、直径、散布矩阵、协方差矩阵

聚类过程中用到类与类之间的距离也称为连接类与类之间的距离包括最短距离、最长距离、中心距离、平均距离。



<1> 层次聚类

假设类别之间存在层次结构，将样本聚到层次化的类中，层次聚类又有聚合/自下而上、分裂/自上而下两种方法。

聚合（自下而上）：聚合法开始将每个样本各自分裂到一个类，之后将相距最近的两类合并，建立一个新的类，重复次操作知道满足停止条件，得到层次化的类别。

分裂（自上而下）： 分裂法开始将所有样本分到一个类，之后将已有类中相距最远的样本分到两个新的类，重复此操作直到满足停止条件，得到层次化的类别


聚合聚类需要预先确定下面三个要素：

（1）距离或相似度； （2）合并规则； （3）停止条件。

根据这些概念的不同组合，就可以得到不同的聚类方法。


<2> Kmeans聚类（基于划分的聚类方法）：

类别数k事先指定
以欧氏距离平方表示样本之间的距离或相似度
以中心或样本的均值表示类别
以样本和其所属类的中心之间的距离的总和为优化的目标函数
得到的类别是平坦的、非层次化的
算法是迭代算法，不能保证得到全局最优

K均值聚类算法，首先选择k个类的中心，将样本分到与中心最近的类中，得到一个聚类结果；
然后计算每个类的样本的均值，作为类的新的中心；
重复以上步骤，直到收敛为止。

k均值聚类是基于中心的聚类方法，通过迭代，将样本分到k个类中，使得每个样本与其所属类的中心或均值最近，得到k个平坦的，非层次化的类别，构成对空间的划分。

"""

import math

class ClusterNode:
    def __init__(self,vec,left=None,right=None,distance=-1,id=None,count=1):
        """
        :param vec: 保存两个数据聚类后形成新的中心
        :param left: 左节点
        :param right:  右节点
        :param distance: 两个节点的距离
        :param id: 用来标记哪些节点是计算过的
        :param count: 这个节点的叶子节点个数
        """
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count

def euler_distance(point1,point2):
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a,b in zip(point1,point2):
        distance += math.pow(a-b,2)
    return math.sqrt(distance)

# 层次聚类 [sklearn.cluster.AgglomerativeClustering]

class Hierarchical:

    def __init__(self,k):
        self.k = k
        self.labels = labels
    
    def fit(self,x):
        nodes = [ClusterNode(vec=v,id=i) for i,v in enumerate(x)]
        # 存储任意两个节点之间的距离
        distances = {}
        point_num,feature_num = x.shape
        self.labels = [-1] * point_num
        currentclustid = -1
        while len(nodes) > self.k:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None
            # 找距离最近的两个类别
            for i in range(nodes_len-1):
                for j in range(i+1,nodes_len):
                    d_key = (nodes[i].id,nodes[j].id)
                    if d_key not in distance:
                        distances[d_key] = euler_distance(nodes[i].vec, nodes[j].vec)
                    d = distance[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i,j)
            # 构建新的类
            part1，part2 = closest_part
            node1, node2 = nodes[part1],nodes[part2]
            # 新类的中心
            new_vec = [ (node1.vec[i] * node1.count + node2.vec[i] * node2.count) / (node1.count + node2.count)
                        for i in range(feature_num)]
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   id=currentclustid,
                                   count=node1.count + node2.count)
            currentclustid -= 1
            del nodes[part1],nodes[part2]
            nodes.append(new_node)
        
        self.nodes = nodes
        self.calc_label()
    
    def calc_label(self):
        """
        调取聚类的结果
        """
        for i, node in enumerate(self.nodes):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)
    
    def leaf_traversal(self,node,label):
        if node.left == None and node.right==None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left,label)
        if node.right:
            self.leaf_traversal(node.right,label)


import math
import random
import numpy as np
from sklearn import datasets,cluster
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = iris['data'][:,:2]

my = Hierarchical(3)
my.fit(data)
labels = np.array(my.labels)
print(labels)


# kmeans

class Kmeans:
    
    def __init__(self,k,n=20):
        self.k = k
        self.n = n
        # max_iter：最大的迭代次数，一般如果是凸数据集的话可以不管这个值，如果数据集不是凸的，可能很难收敛，此时可以指定最大的迭代次数让算法可以及时退出循环。
    
    def fit(self,x,centers=None):
        # 第一步，随机选择 K 个点, 或者指定
        if centers is None:
            idx = np.random.randint(low=0,high=len(x),size=self.k)
            centers = x[idx]
        print(centers)

        inters = 0
        while inters < self.n:
            
            points_set = {key: [] for key in range(self.k)}

            # 第二步，遍历所有点 P, 将 P 放入最近的聚类中心的集合中
            for p in x:
                nearest_index = np.argmin(np.sum((centers-p)**2,axis=1)**0.5)
                points_set[nearest_index].append(p)
            
            # 第三步，遍历每一个点集, 计算新的聚类中心
            for i_k in range(self.k):
                centers[i_k] = sum(points_set[i_k])/len(points_set[i_k])
            inters += 1
        
        return points_set, centers


m = MyKmeans(3)
points_set, centers = m.fit(data)


# using sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, max_iter=100).fit(data)
gt_labels__ = kmeans.labels_
centers__ = kmeans.cluster_centers_

"""
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0,
       2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
       0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2])

array([[6.81276596, 3.07446809],
       [5.006     , 3.428     ],
       [5.77358491, 2.69245283]])
"""

# 寻找 K 值
from sklearn.cluster import KMeans

loss = []

for i in range(1, 10):
    # 该算法的目的是选择出质心，使得各个聚类内部的inertia值最小化
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    kmeans = KMeans(n_clusters=i, max_iter=100).fit(data)
    loss.append(kmeans.inertia_ / len(data) / 3)
