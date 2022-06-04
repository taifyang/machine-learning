import collections
import numpy as np


class ID3:
    '''
    description:    ID3模型
    '''

    def __init__(self, x, y, labels):
        '''
        description:    构造方法
        param self
        param x         特征
        param y         标签
        param labels    特征标签
        '''
        self.x = x
        self.y = y
        self.data = np.hstack((self.x, self.y))
        self.labels = labels
        self.tree = {}

    def calEntropy(self, data):
        '''
        description:    计算信息熵
        param self
        param data      列表等序列
        return          输入数据的信息熵
        '''
        entropy = 0
        c = collections.Counter(np.array(data).ravel())
        total = len(data)
        for i in c.values():
            entropy -= i/total * np.log(i/total)
        return entropy

    def splitdata(self, data, col, value):
        '''
        description:    数据划分
        param self
        param data      列表等序列
        param col       待划分特征列的数字索引
        param value
        return          返回输入data的col列等于value的去掉col列的矩阵
        '''
        data_r = data[data[:, col] == value]
        data_r = np.hstack((data_r[:, :col], data_r[:, col+1:]))
        return data_r

    def getBestFeature(self, data):
        '''
        description:    取得最优特征
        param self
        param data      形式为[x y]的矩阵
        return          最优特征所在列索引
        '''
        entropy_list = []
        numberAll = data.shape[0]
        for col in range(data.shape[1]-1):
            entropy_splited = 0
            for value in np.unique(data[:, col]):
                y_splited = self.splitdata(data, col, value)[:, -1]
                entropy = self.calEntropy(y_splited)
                entropy_splited += len(y_splited)/numberAll*entropy
            entropy_list.append(entropy_splited)
        return entropy_list.index(min(entropy_list))

    def CreateTree(self, data, label):
        '''
        description:    建树
        param self
        param data      形如[x y]的矩阵
        param label     特征
        return          决策树字典
        '''
        feature_label = label.copy()
        if len(np.unique(data[:, -1])) == 1:
            return data[0, -1]
        if data.shape[1] == 1:
            print(collections.Counter(data[:, -1]).most_common()[0][0])
            return collections.Counter(data[:, -1]).most_common()[0][0]
        bestFeature = self.getBestFeature(data)
        bestFeatureLabel = feature_label[bestFeature]
        treeDict = {bestFeatureLabel: {}}

        del feature_label[bestFeature]
        for value in np.unique(data[:, bestFeature]):
            sub_labels = feature_label[:]
            splited_data = self.splitdata(data, bestFeature, value)
            treeDict[bestFeatureLabel][value] = self.CreateTree(
                splited_data, sub_labels)
        return treeDict

    def fit(self):
        '''
        description:    拟合模型
        param self
        '''
        self.tree = self.CreateTree(self.data, self.labels)

    def predict_vec(self, vec, input_tree=None):
        '''
        description:        预测向量
        param self
        param vec           输入向量
        param input_tree    输入树
        return              预测值
        '''
        if input_tree == None:
            input_tree = self.tree

        featureIndex = self.labels.index(list(input_tree.keys())[0])
        secTree = list(input_tree.values())[0]
        vec_feature_val = vec[featureIndex]

        if type(secTree.get(vec_feature_val)) != dict:
            return secTree.get(vec_feature_val)
        else:
            return self.predict_vec(vec, secTree.get(vec_feature_val))

    def predict(self, x):
        '''
        description:    预测
        param self
        param x         特征
        return          预测值
        '''
        out_put = []
        for i in x:
            out_put.append(self.predict_vec(i))
        return out_put


if __name__ == '__main__':
    x = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    y = np.array([[1], [1], [0], [0], [0]])
    labels = [0, 1]
    id3 = ID3(x, y, labels)
    id3.fit()
    print('预测值为：', id3.predict([[1, 0]]))
