import math
import copy
import numpy as np

class EM:
    # 指定k个高斯分布参数，这里指定k=2。注意2个高斯分布具有相同均方差sigma，分别为mu1,mu2。
    def __init__(self, x, sigma, k, n):
        self.x = x
        self.sigma = sigma
        self.k = k
        self.N = n
        self.mu = np.random.random(2)  #随机产生一个初始均值。
        self.expectations = np.zeros((n,k))  #k个高斯分布，100个二维向量组成的矩阵。
        print(self.mu, self.expectations)

    # EM算法：步骤1，计算E[zij]
    def e_step(self):
        #求期望。sigma协方差，k高斯混合模型数，N数据个数。
        for i in range(0, self.N):
            denom = 0
            for j in range(0, self.k):
                denom += math.exp((-1/(2*(float(self.sigma**2))))*(float(self.x[0,i]-self.mu[j]))**2)  #分母项  mu(j)第j个高斯分布的均值。
            for j in range(0,k):
                numer = math.exp((-1/(2*(float(self.sigma**2))))*(float(self.x[0,i]-self.mu[j]))**2)  #分子项
                self.expectations[i,j] = numer / denom      #期望，计算出每一个高斯分布所占的期望，即该高斯分布以多大比例形成这个样本

    # EM算法：步骤2，求最大化E[zij]的参数mu
    def m_step(self):
        for j in range(0, self.k): #遍历k个高斯混合模型数据                   
            numer = 0  #分子项
            denom = 0  #分母项
            for i in range(0, self.N):
                numer += self.expectations[i,j]*self.x[0,i]  #  每一个高斯分布的期望*该样本的值
                denom += self.expectations[i,j]  #第j个高斯分布的总期望值作为分母
                self.mu[j] = numer / denom   #第j个高斯分布新的均值

    # 算法迭代iter_num次，或达到精度epsilon停止迭代
    def predict(self, iter_num, epsilon):
        print(self.x)
        print(u"初始<u1,u2>:", self.mu)   #初始均值
        for i in range(iter_num):
            old_mu = copy.deepcopy(self.mu)  #算法之前的mu
            self.e_step()
            self.m_step()
            print(i,self.mu)  #经过EM算法之后的mu
            if sum(abs(self.mu-old_mu)) < epsilon:
                break


if __name__ == '__main__':
    sigma = 6
    mu1 = 40
    mu2 = 20
    n = 100
    k = 2
    x = np.zeros((1,n)) #x产生的数据 ,k维向
    for i in range(0,n):
        if np.random.random(1) > 0.5:
        #随机从均值为mu1,mu2的分布中取样。
            x[0,i] = np.random.normal()*sigma + mu1
        else:
            x[0,i] = np.random.normal()*sigma + mu2

    em = EM(x, sigma, k, n)
    em.predict(iter_num=100, epsilon=0.001)

  