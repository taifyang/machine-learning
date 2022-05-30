import numpy as np

#Logistic回归模型
class LogisticRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = np.zeros(self.x.shape[1])
        self.b = 0

    def Logistic_sigmoid(self, y):
        #非线性层，将值域空间映射为（0，1）
        return np.exp(y)/(1+np.exp(y))

    def Logistic_cost(self, p, y):
        #损失函数
        return np.sum(-y*np.log(p)-(1-y)*np.log(1-p))

    def Logistic_BP(self, alpha, iters):
        #反向传播函数
        for i in range(iters):
            p = np.dot(self.x, self.w.T)+self.b
            a = self.Logistic_sigmoid(p)
            print('iters:', i,' cost:',  self.Logistic_cost(a, y))
            dz = a -self.y
            self.w -= alpha*np.dot(dz.T, self.x)
            self.b -= alpha*sum(dz)
        return self.w, self.b

    def Logistic_predict(self, x):
        #预测函数
        return self.Logistic_sigmoid(np.dot(x, self.w.T)+self.b)


if __name__ == '__main__':
    x = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    lg = LogisticRegression(x, y)
    w, b = lg.Logistic_BP(alpha=0.1, iters=100)
    print('最终训练得到的w和b为：', w, ',', b)
    pre = lg.Logistic_predict(np.array([[2.9]]))
    print('预测结果为：', pre)
