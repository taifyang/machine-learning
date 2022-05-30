import random
import pandas as pd
import numpy as np

class KMeans:
    def __init__(self, dataSet, k):
        self.dataSet = dataSet
        self.k = k

    # 计算欧拉距离
    def calcDis(self, centroids):
        clalist=[]
        for data in self.dataSet:
            diff = np.tile(data, (self.k, 1)) - centroids  #相减  a=[0,1,2], (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
            squaredDiff = diff ** 2     #平方
            squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
            distance = squaredDist ** 0.5  #开根号
            clalist.append(distance) 
        clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
        return clalist

    # 计算质心
    def classify(self, centroids):
        # 计算样本到质心的距离
        clalist = self.calcDis(centroids)
        #print(dataSet, centroids, clalist)
        # 分组并计算新的质心
        minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
        #print(clalist, minDistIndices)
        newCentroids = pd.DataFrame(self.dataSet).groupby(minDistIndices).mean() #DataFrame(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
        #print(newCentroids, newCentroids.values)
        newCentroids = newCentroids.values    
        changed = newCentroids - centroids #计算变化量
        return changed, newCentroids

    # 使用k-means分类
    def predict(self):
        # 随机取质心
        #centroids = random.sample(self.dataSet, self.k)  
        centroids = self.dataSet[np.random.choice(self.dataSet.shape[0], size=self.k, replace=False), :]
        # 更新质心 直到变化量全为0
        changed, newCentroids = self.classify(centroids)    
        #print(centroids,newCentroids) 
        while np.any(changed != 0):          
            changed, newCentroids = self.classify(newCentroids)
            #print(changed)   
        centroids = newCentroids.tolist()   #tolist()将矩阵转换成列表
        # 根据质心计算每个集群
        cluster = []
        clalist = self.calcDis(centroids) #调用欧拉距离
        minDistIndices = np.argmin(clalist, axis=1)  
        for i in range(self.k):
            cluster.append([])
        for i, j in enumerate(minDistIndices):   #enumerate()可同时遍历索引和遍历元素
            cluster[j].append(self.dataSet[i])        
        return centroids, cluster
 

if __name__=='__main__': 
    x = np.array([[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]])
    kmeans = KMeans(x, 2)
    centroids, cluster = kmeans.predict()
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)

