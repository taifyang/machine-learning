import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    x = np.array([[0], [1], [2], [3]])
    y = np.array([[0], [0], [1], [1]])
    clf = LogisticRegression()
    clf.fit(x, y)
    w = clf.coef_
    b = clf.intercept_
    print('最终训练得到的w和b为：', w, ',', b)
    print('预测结果为：', clf.predict_proba([[2.9]]))
