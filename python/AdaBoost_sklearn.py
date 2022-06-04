import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor


if __name__ == '__main__':
    x = np.array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    reg = AdaBoostRegressor(tree.DecisionTreeRegressor(
        max_depth=4), n_estimators=300, random_state=np.random.RandomState(1))
    reg.fit(x, y)
    print('预测值为：', reg.predict([[0, 0]]))
