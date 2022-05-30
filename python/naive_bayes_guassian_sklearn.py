import numpy as np
from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':   
    x = np.array([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],[1,2,2,1,1,1,2,2,3,3,3,2,2,3,3]]).T
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    gnb = GaussianNB()
    gnb.fit(x,y)
    print('预测值为：', gnb.predict(np.array([[2,1]])))
