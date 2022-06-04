import math
import copy
import numpy as np


class EM:
    '''
    description:    EM模型
    '''

    def __init__(self, x, sigma, k, n):
        '''
        description:     构造方法
        param self
        param x          特征
        param sigma      高斯分布均方差
        param k          高斯混合模型数
        param n          数据个数
        '''
        self.x = x
        self.sigma = sigma
        self.k = k
        self.N = n
        self.mu = np.random.random(2)
        self.expectations = np.zeros((n, k))

    def e_step(self):
        '''
        description:    EM算法步骤1，计算E[zij]
        param self
        '''
        for i in range(0, self.N):
            denom = 0
            for j in range(0, self.k):
                denom += math.exp((-1/(2*(float(self.sigma**2)))) * (float(self.x[0, i]-self.mu[j]))**2)
            for j in range(0, k):
                numer = math.exp((-1/(2*(float(self.sigma**2)))) * (float(self.x[0, i]-self.mu[j]))**2)
                self.expectations[i, j] = numer / denom

    def m_step(self):
        '''
        description:    EM算法步骤2，求最大化E[zij]的参数mu
        param self
        '''
        for j in range(0, self.k):
            numer = 0
            denom = 0
            for i in range(0, self.N):
                numer += self.expectations[i, j]*self.x[0, i]
                denom += self.expectations[i, j]
                self.mu[j] = numer / denom

    def predict(self, iter_num, epsilon):
        '''
        description:    预测
        param self
        param iter_num  迭代次数
        param epsilon   精度
        '''
        for i in range(iter_num):
            old_mu = copy.deepcopy(self.mu)
            self.e_step()
            self.m_step()
            print(i, self.mu)
            if sum(abs(self.mu-old_mu)) < epsilon:
                break


if __name__ == '__main__':
    sigma = 6
    mu1 = 40
    mu2 = 20
    n = 100
    k = 2
    x = np.zeros((1, n))
    for i in range(0, n):
        if np.random.random(1) > 0.5:
            x[0, i] = np.random.normal()*sigma + mu1
        else:
            x[0, i] = np.random.normal()*sigma + mu2
    em = EM(x, sigma, k, n)
    em.predict(iter_num=100, epsilon=0.001)
