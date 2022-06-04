import random
import pandas as pd
import numpy as np


class KMeans:
    '''
    description:    KMeans模型
    '''

    def __init__(self, dataSet, k):
        '''
        description:    构造方法
        param self
        param dataSet   数据
        param k         聚类数目
        '''
        self.dataSet = dataSet
        self.k = k

    def calcDis(self, centroids):
        '''
        description:    计算欧拉距离
        param self
        param centroids 聚类中心
        return          返回一个每个点到质点的距离
        '''
        clalist = []
        for data in self.dataSet:
            diff = np.tile(data, (self.k, 1)) - centroids
            squaredDiff = diff ** 2
            squaredDist = np.sum(squaredDiff, axis=1)
            distance = squaredDist ** 0.5
            clalist.append(distance)
        clalist = np.array(clalist)
        return clalist

    def classify(self, centroids):
        '''
        description:    计算质心
        param self
        param centroids 质心
        return          改变量，新质心
        '''
        clalist = self.calcDis(centroids)
        minDistIndices = np.argmin(clalist, axis=1)
        newCentroids = pd.DataFrame(self.dataSet).groupby(minDistIndices).mean()
        newCentroids = newCentroids.values
        changed = newCentroids - centroids
        return changed, newCentroids

    def predict(self):
        '''
        description:    预测
        param self
        return          质心，聚类
        '''
        centroids = self.dataSet[np.random.choice(self.dataSet.shape[0], size=self.k, replace=False), :]
        changed, newCentroids = self.classify(centroids)
        while np.any(changed != 0):
            changed, newCentroids = self.classify(newCentroids)
        centroids = newCentroids.tolist()
        cluster = []
        clalist = self.calcDis(centroids)
        minDistIndices = np.argmin(clalist, axis=1)
        for i in range(self.k):
            cluster.append([])
        for i, j in enumerate(minDistIndices):
            cluster[j].append(self.dataSet[i])
        return centroids, cluster


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]])
    kmeans = KMeans(x, 2)
    centroids, cluster = kmeans.predict()
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
