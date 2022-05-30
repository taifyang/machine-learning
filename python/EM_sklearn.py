from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


if __name__ == '__main__':
    x, y = make_blobs(n_samples=100, n_features=1, centers=[[40],[20]], cluster_std=6)  #产生实验数据
    gmm = GaussianMixture(n_components=2, max_iter=100)
    gmm.fit(x)
    print(gmm.means_)


