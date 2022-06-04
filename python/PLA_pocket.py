import numpy as np
import random


class Pocket:
    '''
    description:    Pocket模型
    '''

    def __init__(self, x, y):
        '''
        description:    构造函数
        param self
        param x         特征
        param y         标签
        '''
        self.x = x
        self.y = y
        self.w = np.zeros(x.shape[1])
        self.best_w = np.zeros(x.shape[1])
        self.b = 0
        self.best_b = 0

    def sign(self, w, b, x):
        '''
        description:    计算y
        param self
        param w         权重
        param b         偏差
        param x         x
        return          y
        '''
        y = np.dot(x, w) + b
        return int(y)

    def classify(self, w, b):
        '''
        description:    分类
        param self
        param w         权重
        param b         偏差
        return          误分类值
        '''
        mistakes = []
        for i in range(self.x.shape[0]):
            tmpY = self.sign(w, b, self.x[i, :])
            if tmpY * self.y[i] <= 0:
                mistakes.append(i)
        return mistakes

    def update(self, label_i, data_i):
        '''
        description:    更新权重
        param self
        param label_i   标签
        param data_i    数据
        '''
        tmp = label_i * data_i
        tmpw = tmp + self.w
        tmpb = self.b + label_i
        if(len(self.classify(self.best_w, self.best_b)) >= (len(self.classify(tmpw, tmpb)))):
            self.best_w = tmp + self.w
            self.best_b = self.b + label_i
        self.w = tmp + self.w
        self.b = self.b + label_i

    def train(self, max_iters):
        '''
        description:        训练
        param self
        param max_iters     最大迭代次数
        return              权重和偏差
        '''
        iters = 0
        isFind = False
        while not isFind:
            mistakes = self.classify(self.w, self.b)
            if(len(mistakes) == 0):
                return self.best_w, self.best_b
            n = mistakes[random.randint(0, len(mistakes)-1)]
            self.update(self.y[n], self.x[n, :])
            iters += 1
            if iters == max_iters:
                isFind = True
        return self.best_w, self.best_b


if __name__ == '__main__':
    x = np.array([[3, -3], [4, -3], [1, 1], [1, 2]])
    y = np.array([-1, -1, 1, 1])
    myPocket_PLA = Pocket(x, y)
    w, b = myPocket_PLA.train(100)
    print('最终训练得到的w和b为：', w, b)
