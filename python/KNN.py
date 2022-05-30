import numpy as np
import operator

class KNN():
    def __init__(self, x, y, k, p):
        self.k = k
        self.p = p
        self.x = x
        self.y = y

    def predict(self, x):    
        diff = np.tile(x, (self.x.shape[0], 1)) - self.x  #计算预测数据和训练数据的差值
        dist=np.linalg.norm(diff, ord=self.p, axis=1, keepdims=False) #计算范数
        dist_sorted=dist.argsort() #返回从小到大排序的索引

        #分类投票
        count = {}
        for i in range(self.k):
            vote = self.y[dist_sorted[i]]
            count[vote] = count.get(vote, 0) + 1

        #对分类投票数从低到高进行排序
        count_sorted = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
        return count_sorted[0][0]
      

if __name__=='__main__':
    x=np.array([[0, 10], [1, 8], [10, 1], [7, 4]])
    y=np.array([0, 0, 1, 1])
    knn=KNN(x, y, 3, 2)
    print("预测值为：",knn.predict(np.array([[6, 2]])))