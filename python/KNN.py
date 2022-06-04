import numpy as np
import operator


class KNN():
    '''
    description:    KNN模型
    '''

    def __init__(self, x, y, k, p):
        '''
        description:    构造方法
        param self
        param x         特征
        param y         标签
        param k         邻近数
        param p         度量方式
        '''
        self.k = k
        self.p = p
        self.x = x
        self.y = y

    def predict(self, x):
        '''
        description:    预测
        param self
        param x         输入特征
        return
        '''
        diff = np.tile(x, (self.x.shape[0], 1)) - self.x
        dist = np.linalg.norm(diff, ord=self.p, axis=1, keepdims=False)
        dist_sorted = dist.argsort()

        count = {}
        for i in range(self.k):
            vote = self.y[dist_sorted[i]]
            count[vote] = count.get(vote, 0) + 1

        count_sorted = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
        return count_sorted[0][0]


if __name__ == '__main__':
    x = np.array([[0, 10], [1, 8], [10, 1], [7, 4]])
    y = np.array([0, 0, 1, 1])
    knn = KNN(x, y, 3, 2)
    print("预测值为：", knn.predict(np.array([[6, 2]])))
