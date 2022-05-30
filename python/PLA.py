import numpy as np

#感知机模型
class Perceptron:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.w=np.zeros(self.x.shape[1]) #初始化权重w
        self.b = 0

    def sign(self, w, b, x):
        y = np.dot(x, w)+b
        return int(y)

    def update(self,label_i,data_i):  
        tmp = label_i*data_i
        self.w += tmp
        self.b += label_i

    def train(self):
        isFind=False
        while not isFind:
            count=0
            for i in range(self.x.shape[0]):
                tmp_y=self.sign(self.w, self.b, self.x[i,:])
                if tmp_y*self.y[i]<=0:   #如果误分类
                    count+=1
                    self.update(self.y[i],self.x[i,:])
            if count==0:
                isFind=True
        return self.w, self.b
        

if __name__ == '__main__':
    x = np.array([[3,-3],[4,-3],[1,1],[1,2]])
    y = np.array([-1, -1, 1, 1])
    myperceptron = Perceptron(x, y)
    w, b = myperceptron.train()
    print('最终训练得到的w和b为：', w, b)