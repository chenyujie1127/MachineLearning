"""
SVM: 支持向量机

"""
class SVM:
    def __init__(Self,max_iter=100,kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel
    def init_args(self,features,labels):
        self.m,self.n=features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0

    def _KKT(self,i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i]==0:
            return y_g >= 1
        elif 0<self.alpha[i]<self.C:
            return y_g == 1
        else:
            return y_g <= 1
    
    # g(x)预测值，输入xi（X[i]）【非线性支持向量机】
    def _g(self,i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j]*self.kernel(self.X[i],self.X[j])
        return r

    def kernel(self,x1,x2):
        if self._kernel == 'linear':
            return sum([ x1[k]*x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([ x1[k]*x2[k] for k in range(self.n)])+1)**2
        return 0
    
    # E（x）为g(x)对输入x的预测值和y的差
    def _E(self,i):
        return self._g(i)-self.Y[i]

    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 不满足的列表
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        # 不满足的加在满足的后面
        index_list.extend(non_satisfy_list)
        
        for i in index_list:
            if self._KKT(i):
                continue
            # 看差值
            # 当|E1-E2|越大时，优化后的α1、α2改变越大。所以，如果E1是正的，那么E2越负越好，如果E1是负的，那么E2越正越好
            E1 = self.E[i]
            if E1 >= 0:
                j = min(range(self.m),key=lambda x:self.E[x])
            else:
                j = max(range(self.m),key=lambda x:self.E[x])
            return i,j

    def _compare(self,_alpha,L,H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha
    
    # SMO算法实现【序列最小优化算法】
    # https://blog.csdn.net/wds2006sdo/article/details/53156589
    
    """
    SMO算法是支持向量机学习的一种快速算法，其特点是不断地将原二次规划问题分解为只有两个变量的二次规划子问题，
    并对子问题进行解析求解，直到所有变量满足KKT条件为止。
    这样通过启发式的方法得到原二次规划问题的最优解。
    因为子问题有解析解，所以每次计算子问题都很快，虽然计算子问题次数很多，但在总体上还是高效的。
    【计算出一组最优的alpha和常数项b的值。
    SMO算法的中心思想就是每次选出两个alpha进行优化（
    之所以是两个是因为alpha的约束条件决定了其与标签乘积的累加等于0，因此必须一次同时优化两个，否则就会破坏约束条件），然后固定其他的alpha值】
    """
    def fit(self,features,labels):
        
        self.init_args(features,labels)
        
        for t in range(max_iter):
            i1, i2 = self._init_alpha()

            if self.Y[i1] == self.Y[i2]:
                L = max(0,self.alpha[i1]+self.alpha[i2]-self.C)
                H = min(self.C,self.alpha[i1]+self.alpha[i2])
            else:
                L = max(0,self.alpha[i2]-self.alpha[i1])
                H = min(self.C,self.C+self.alpha[i2]-self.alpha[i1])
        
            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(
                self.X[i2],
                self.X[i2]) - 2 * self.kernel(self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue
            
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (
                E1 - E2) / eta  #此处有修改，根据书上应该是E1 - E2，书上130-131页
            alpha2_new = self._compare(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (
                self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (
                alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                    self.X[i2],
                    self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (
                alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                    self.X[i2],
                    self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
        
        return 'train done!'
            
    
    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)
    
     def _weight(self):
        # linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w

svm = SVM(max_iter=200)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
