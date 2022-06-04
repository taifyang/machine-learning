import numpy as np
from sklearn.linear_model import Perceptron


if __name__ == '__main__':
    x = np.array([[3, -3], [4, -3], [1, 1], [1, 2]])
    y = np.array([-1, -1, 1, 1])
    clf = Perceptron(fit_intercept=True, max_iter=100, shuffle=False)  # 定义感知机
    clf.fit(x, y)
    w = clf.coef_
    b = clf.intercept_
    print('最终训练得到的w和b为：', w, ',', b)
