import numpy as np


class LogisticRegression:
    '''
    description:    逻辑斯蒂回归模型
    '''

    def __init__(self, x, y):
        '''
        description:    构造方法
        param self
        param x         特征
        param y         标签
        '''
        self.x = x
        self.y = y
        self.w = np.zeros(self.x.shape[1])
        self.b = 0

    def Logistic_sigmoid(self, y):
        '''
        description:    非线性层，将值域空间映射为(0, 1)
        param self
        param y         标签
        return          映射
        '''
        return np.exp(y)/(1+np.exp(y))

    def Logistic_cost(self, p, y):
        '''
        description:    损失函数
        param self
        param p         概率
        param y         标签
        return          损失
        '''
        return np.sum(-y*np.log(p)-(1-y)*np.log(1-p))

    def Logistic_BP(self, alpha, iters):
        '''
        description:    反向传播函数
        param self
        param alpha     学习率
        param iters     迭代次数
        return
        '''
        for i in range(iters):
            p = np.dot(self.x, self.w.T)+self.b
            a = self.Logistic_sigmoid(p)
            print('iters:', i, ' cost:',  self.Logistic_cost(a, y))
            dz = a - self.y
            self.w -= alpha*np.dot(dz.T, self.x)
            self.b -= alpha*sum(dz)
        return self.w, self.b

    def Logistic_predict(self, x):
        '''
        description:    预测
        param self
        param x         特征
        return          预测值
        '''
        return self.Logistic_sigmoid(np.dot(x, self.w.T)+self.b)


if __name__ == '__main__':
    x = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    lg = LogisticRegression(x, y)
    w, b = lg.Logistic_BP(alpha=0.1, iters=100)
    print('最终训练得到的w和b为：', w, ',', b)
    pre = lg.Logistic_predict(np.array([[2.9]]))
    print('预测结果为：', pre)
