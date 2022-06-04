import numpy as np


class GaussionNB:
    '''
    description:    高斯朴素贝叶斯模型
    '''

    def __init__(self, fit_prior=True):
        '''
        description:    构造方法
        param self
        param fit_prior 是否学习类的先验几率，False则使用统一的先验
        '''
        self.fit_prior = fit_prior
        self.class_prior = None
        self.classes = None
        self.mean = None
        self.var = None
        self.predict_prob = None

    def fit(self, x, y):
        '''
        description:    建立模型
        param self
        param x         特征
        param y         标签
        '''
        self.classes = np.unique(y)
        if self.class_prior == None:
            class_num = len(self.classes)
            self.class_prior = {}
            if not self.fit_prior:
                for d in self.classes:
                    self.class_prior[d] = 1.0/class_num
            else:
                self.class_prior = {}
                for d in self.classes:
                    c_num = np.sum(np.equal(y, d))
                    self.class_prior[d] = c_num / len(y)

        self.mean = {}
        self.var = {}
        y = list(y)
        for yy in self.class_prior.keys():
            y_index = [i for i, label in enumerate(y) if label == yy]
            for i in range(len(x)):
                x_class = []
                for j in y_index:
                    x_class.append(x[i][j])
                    pkey = str(i) + '|' + str(yy)
                    mean = np.mean(x_class)
                    var = np.var(x_class)
                    self.mean[pkey] = mean
                    self.var[pkey] = var
        return self

    def _calculat_prob_gaussion(self, mu, sigma, x):
        '''
        description:    计算高斯概率
        param self
        param mu        均值
        param sigma     方差
        param x         特征
        return          高斯概率
        '''
        return 1.0/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))

    def predict(self, x):
        '''
        description:    预测
        param self
        param x         特征
        return          预测值
        '''
        labels = []
        for i in range(x.shape[0]):
            self.predict_prob = {}
            for yy in self.class_prior.keys():
                self.predict_prob[yy] = self.class_prior[yy]
                for c, d in enumerate(list(x[i])):
                    tkey = str(c)+'|'+str(yy)
                    mu = self.mean[tkey]
                    sigma = self.var[tkey]
                    self.predict_prob[yy] = self.predict_prob[yy] * \
                        self._calculat_prob_gaussion(mu, sigma, d)
            label = max(self.predict_prob, key=self.predict_prob.get)
            labels.append(label)
        return labels


if __name__ == '__main__':
    x = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                 [1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 3]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    gnb = GaussionNB()
    gnb.fit(x, y)
    print('预测值为：', gnb.predict(np.array([[2, 1]])))
