import numpy as np
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    x = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 2.9, 4.1])
    clf = LinearRegression()
    clf.fit(x, y)
    w = clf.coef_
    b = clf.intercept_
    print('最终训练得到的w和b为：', w, ',', b)
