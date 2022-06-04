import numpy as np
from sklearn import tree


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    y = np.array([[1], [1], [0], [0], [0]])
    clf = tree.DecisionTreeRegressor(max_depth=4)
    clf.fit(x, y)
    print('预测值为：', clf.predict([[1, 0]]))

