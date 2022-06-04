import numpy as np
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    x = np.array([[0, 10], [1, 8], [10, 1], [7, 4]])
    y = np.array([0, 0, 1, 1])
    knn = KNeighborsClassifier(n_neighbors=3, p=2)
    knn.fit(x, y)
    print("预测值为：", knn.predict(np.array([[6, 2]])))
