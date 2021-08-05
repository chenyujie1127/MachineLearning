"""
KNN分类算法
"""
import numpy as np
# 解析文本
"""
每行代表一个人的特征
前三列分别对应不同的特征【每年获得的飞行常客里程数 ｜ 玩视频游戏所耗时间百分比 ｜ 每周消费的冰淇淋公升数】
最后一列表示不同男人类型 【不喜欢的人 ｜ 魅力一般的人 ｜ 极具魅力的人】

40920	8.326976	0.953952	3
14488	7.153469	1.673904	2
26052	1.441871	0.805124	1
75136	13.147394	0.428964	1
38344	1.669788	0.134296	1

可用前四行存入data.txt文本
最后一行用作测试数据data

"""

def file2matrix(filename):
    feature_matrix = []
    labels = []
    with open(filename) as f:
        for line in f:
            data = line.strip().split('\t')
            label = data[-1]
            feature_matrix.append(list(map(float,data[0:-1])))
            labels.append(int(label))
    feature_matrix = np.array(feature_matrix)
    labels = np.array(labels)
    return feature_matrix,labels


def autoNorm(feature):
    # 归一化特征值，消除属性之间量级不同导致的影响
    # Y = (X-X_min)/(X_max-X_min)
    # 其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    # 按列取最大值和最小值
    minVal = feature.min(0)
    maxVal = feature.max(0)
    # print(minVal)
    # print(maxVal)
    normFeature = (feature-minVal) / (maxVal - minVal)
    return normFeature




def KNN(data,feature,labels,k):
    # data: 用于分类的一个样本
    # 多少样本
    size = feature.shape[0]

    # 计算欧式距离,(A1-A2)^2+(B1-B2)^2+(c1-c2)^2 ,相减->平方->行相加->开方
    # 计算当前样本到其他样本的距离，tile是对当前样本进行复制
    # 直接平方,再按行相加
    
    distance = np.tile(data,(size,1))-feature
    squareDis = distance ** 2
    sumDistance = squareDis.sum(1)
    squareDis = sumDistance ** 0.5

    # 对距离进行排序,得到距离从小到大的坐标:将x中的元素从小到大排列，提取其对应的index(索引)
    # 再找到这k个特征所对应的标签，记录最大的label数字
    sortedDis = squareDis.argsort()
    

    classCount = {}

    for i in range(k):
        votelabel = labels[sortedDis[i]]
        classCount[votelabel] = classCount.get(votelabel,0)+1
    
    # 取最大的label返回
    maxClassCount = max(classCount.values())
    return maxClassCount

    """
    another method: 使用了numpy broadcasting:
    计算距离, 找k个最近样本label的方法
    
    dis = np.sum((data-feature)**2,axis = 1)**0.5
    k_labels = [labels[index] for index in dis.argsort()[0:k]]
    import collections
    # 结果不一样的原因，很可能是因为value值一样
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label
    """
    

# 测试数据
data = np.array([38344,1.669788,0.134296])
# 数据归一化
data = autoNorm(data)
# 读取训练样本数据
feature_matrix,labels = file2matrix('data.txt')
# 归一化训练样本特征数据
normFeature = autoNorm(feature_matrix)
# 送入测试数据与训练样本数据进行比较，得到当前样本属于哪个标签
result = KNN(data,feature_matrix,labels,3)
print(result)
