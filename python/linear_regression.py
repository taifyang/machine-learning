import numpy as np


def LinearRegression(x, y, alpha, iters):
    '''
    description:    线性回归模型
    param x         特征
    param y         标签
    param alpha     学习率
    param iters     迭代次数
    return          权重和偏置
    '''
    x = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    y = y.T
    w = np.mat(np.zeros(x.shape[1]))
    for i in range(iters):
        print('iters:', i, ' cost:', 1 / (2 * y.size)  * np.sum(np.power(x * w.T - y, 2)))
        w -= (alpha / y.size) * (x * w.T - y).T * x
    return np.array(w)[0][0], np.array(w)[0][1:]


if __name__ == '__main__':
    x = np.array([[1], [2], [3], [4]])
    y = np.array([[1, 2, 2.9, 4.1]])
    w, b = LinearRegression(x, y, alpha=0.1, iters=100)
    print('最终训练得到的w和b为：', w, b)
