import numpy as np

class GaussionNB(object):   #计算条件几率的方法不同
    def __init__(self, fit_prior=True):
        self.fit_prior = fit_prior #是否学习类的先验几率，False则使用统一的先验
        self.class_prior = None    #类的先验几率，若指定则先验不能根据数据调整  
        self.classes = None
        self.mean = None
        self.var = None
        self.predict_prob = None

    def fit(self,x,y):
        #计算类别y的先验几率
        self.classes = np.unique(y)
        if self.class_prior == None:#先验几率没有指定
            class_num = len(self.classes)
            self.class_prior = {}
            if not self.fit_prior:
                for d in self.classes:
                    self.class_prior[d] = 1.0/class_num 
            else:
                self.class_prior = {}
                for d in self.classes:
                    c_num = np.sum(np.equal(y,d))
                    self.class_prior[d]= c_num / len(y)
        #print(self.class_prior)           
        #计算条件几率------高斯模型
        self.mean = {}
        self.var = {}
        y = list(y)
        for yy in self.class_prior.keys():
            y_index = [i for i,label in enumerate(y) if label == yy] #标签的先验几率
            #print(y_index)
            for i in range(len(x)):
                x_class =[]
                for j in y_index:
                    x_class.append(x[i][j])
                    pkey = str(i) + '|' + str(yy)
                    mean = np.mean(x_class)
                    var = np.var(x_class)
                    self.mean[pkey] = mean
                    self.var[pkey] = var
                #print(x_class)
        #print(self.mean, self.var)
        return self

    def _calculat_prob_gaussion(self,mu,sigma,x):    
        return  1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2)) 

    def predict(self,x):
        labels = []
        for i in range(x.shape[0]):
            self.predict_prob = {}
            for yy in self.class_prior.keys():
                self.predict_prob[yy] = self.class_prior[yy]
                for c,d in  enumerate(list(x[i])):
                    tkey = str(c)+'|'+str(yy)
                    mu = self.mean[tkey] 
                    sigma = self.var[tkey] 
                    #print(mu,sigma)
                    self.predict_prob[yy] = self.predict_prob[yy]*self._calculat_prob_gaussion(mu,sigma,d)     
            #print(self.predict_prob)
            label = max(self.predict_prob, key=self.predict_prob.get)
            labels.append(label)
        return labels


if __name__ == '__main__':
    x = np.array([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],[1,2,2,1,1,1,2,2,3,3,3,2,2,3,3]])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    gnb = GaussionNB()
    gnb.fit(x, y)
    print('预测值为：', gnb.predict(np.array([[2,1]])))
