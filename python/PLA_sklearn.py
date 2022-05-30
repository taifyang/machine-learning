import numpy as np
from sklearn.linear_model import Perceptron

#感知机模型
def MyPerceptron(x, y):   
    clf = Perceptron(fit_intercept=True, max_iter=50, shuffle=False) #定义感知机  
    clf.fit(x, y)  #训练感知机   
    w = clf.coef_ #得到权重矩阵    
    b = clf.intercept_ #得到截距
    return w, b


if __name__ == '__main__':
    x = np.array([[3, -3], [4, -3], [1, 1], [1, 2]])
    y = np.array([-1, -1, 1, 1])
    w, b = MyPerceptron(x, y)
    print('最终训练得到的w和b为：', w, ',', b)
