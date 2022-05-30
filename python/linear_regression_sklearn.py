import numpy as np
from sklearn.linear_model import LinearRegression

#线性回归模型
def MyLinearRegression(x, y):   
    clf = LinearRegression()
    clf.fit(x, y)  #训练感知机   
    w = clf.coef_ #得到权重矩阵    
    b = clf.intercept_ #得到截距
    return w, b


if __name__ == '__main__':
    x = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 2.9, 4.1]) 
    w, b = MyLinearRegression(x, y)
    print('最终训练得到的w和b为：', w, ',', b)   
