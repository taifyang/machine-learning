import numpy as np
from sklearn.cluster import KMeans


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]])
    k_means = KMeans(n_clusters=2)
    k_means.fit(x)
    y_predict = k_means.predict(x)
    print(k_means.predict((x[:,:])))
    print(k_means.cluster_centers_)
