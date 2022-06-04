import numpy as np
from sklearn.naive_bayes import BernoulliNB


if __name__ == '__main__':
    x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [
                 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    bnb = BernoulliNB()
    bnb.fit(x, y)
    print('预测值为：', bnb.predict(np.array([[0, 0]])))
