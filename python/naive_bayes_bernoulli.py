import numpy as np


class BernoulliNB:
    '''
    description:    伯努利朴素贝叶斯模型
    '''

    def __init__(self, alpha=1.0, fit_prior=True):
        '''
        description:    构造方法
        param alpha     平滑系数
        param fit_prior 是否学习类的先验几率，False则使用统一的先验
        '''
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = None
        self.classes = None
        self.conditional_prob = None
        self.predict_prob = None

    def fit(self, x, y):
        '''
        description:    计算类别y的先验几率
        param x         数据
        param y         类别
        '''
        self.classes = np.unique(y)
        if self.class_prior == None:
            class_num = len(self.classes)
            self.class_prior = {}
            if not self.fit_prior:
                for d in self.classes:
                    self.class_prior[d] = 1.0/class_num
            else:
                for d in self.classes:
                    c_num = np.sum(np.equal(y, d))
                    self.class_prior[d] = (
                        c_num+self.alpha) / (len(y) + class_num * self.alpha)

        self.conditional_prob = {}
        x_class = [0, 1]
        y = list(y)
        for yy in self.class_prior.keys():
            y_index = [i for i, label in enumerate(y) if label == yy]
            for i in range(len(x)):
                for c in x_class:
                    x_index = [j for j, feature in enumerate(
                        x[i]) if feature == c]
                    xy_count = len(set(x_index) & set(y_index))
                    pkey = str(c) + '|' + str(yy)
                    self.conditional_prob[pkey] = (
                        xy_count+self.alpha) / (len(y_index)+2)

        return self

    def predict(self, x):
        '''
        description:    预测
        param x         数据
        return          预测值
        '''
        labels = []
        for i in range(x.shape[0]):
            self.predict_prob = {}
            for j in self.classes:
                self.predict_prob[j] = self.class_prior[j]
                for d in x[i]:
                    self.predict_prob[j] = self.predict_prob[j] * \
                        self.conditional_prob[str(d) + '|' + str(j)]
            label = max(self.predict_prob, key=self.predict_prob.get)
            labels.append(label)
        return labels


if __name__ == '__main__':
    x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                 [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    bnb = BernoulliNB(alpha=1.0, fit_prior=True)
    bnb.fit(x, y)
    print('预测值为：', bnb.predict(np.array([[0, 0]])))
