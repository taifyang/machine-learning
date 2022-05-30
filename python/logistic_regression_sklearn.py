import numpy as np
from sklearn.linear_model import LogisticRegression

#Logistic回归模型
class MyLogisticRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y  
        self.clf = LogisticRegression()    
    
    def train(self):
        self.clf.fit(self.x, self.y)   
        w = self.clf.coef_
        b = self.clf.intercept_ 
        return w, b

    def predict(self, x):
        return self.clf.predict_proba(x)
        

if __name__ == '__main__':
    x = np.array([[0], [1], [2], [3]])
    y = np.array([[0], [0], [1], [1]])
    lr = MyLogisticRegression(x, y)

    w, b = lr.train()
    print('最终训练得到的w和b为：', w, ',', b)   
    print('预测结果为：', lr.predict([[2.9]]))
