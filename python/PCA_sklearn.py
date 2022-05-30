import numpy as np
from sklearn.decomposition import PCA


if __name__ == '__main__':
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components='mle')
    pca.fit(x)
    print(pca.explained_variance_)
