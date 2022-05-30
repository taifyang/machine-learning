import collections
import numpy as np

class ID3:
    def __init__(self, x, y, labels):
        self.x = x
        self.y = y
        self.data = np.hstack((self.x, self.y))
        self.labels = labels
        self.tree = {}
 
    def calEntropy(self, data):
        '''
        计算信息熵
        :param data:列表等序列
        :return:输入数据的信息熵
        '''
        entropy = 0
        c = collections.Counter(np.array(data).ravel())
        total = len(data)
        for i in c.values():
            entropy -= i/total * np.log(i/total)
        return entropy
 
    def splitdata(self, data, col, value):
        '''
        :param data:
        :param col:待划分特征列的数字索引
        :param value:
        :return:返回输入data的col列等于value的去掉col列的矩阵
        '''       
        data_r = data[data[:, col] == value]
        data_r = np.hstack((data_r[:, :col], data_r[:, col+1:]))
        #print(data, col,value,data_r)
        return data_r
 
    def getBestFeature(self, data):
        '''
        :param data: 形式为[x y]的矩阵
        :return: 最优特征所在列索引
        '''
        entropy_list = []
        numberAll = data.shape[0]
        #print('data',data)
        for col in range(data.shape[1]-1):
            entropy_splited = 0
            #print(data[:, col], np.unique(data[:, col]))
            for value in np.unique(data[:, col]):            
                y_splited = self.splitdata(data, col, value)[:, -1]
                #print(col,value,data)
                entropy = self.calEntropy(y_splited)
                entropy_splited += len(y_splited)/numberAll*entropy
            entropy_list.append(entropy_splited)
        #print(entropy_list)
        return entropy_list.index(min(entropy_list))
 
    def CreateTree(self, data, label):
        '''
        :param data: 形如[x y]的矩阵
        :param feature_label:
        :return: 决策树字典
        '''
        #print(np.unique(data[:, -1]))
        feature_label = label.copy()
        if len(np.unique(data[:, -1])) == 1:
            #print(data[0, -1])
            return data[0, -1]
        if data.shape[1] == 1:
            print(collections.Counter(data[:, -1]).most_common()[0][0])
            return collections.Counter(data[:, -1]).most_common()[0][0]
        bestFeature = self.getBestFeature(data)
        bestFeatureLabel = feature_label[bestFeature]
        #print(bestFeature, bestFeatureLabel)
        treeDict = {bestFeatureLabel: {}}
 
        del feature_label[bestFeature]
        #print(np.unique(data[:, bestFeature]))
        for value in np.unique(data[:, bestFeature]):
            sub_labels = feature_label[:]
            #print(feature_label,sub_labels)
            splited_data = self.splitdata(data, bestFeature, value)     
            #print(splited_data)     
            treeDict[bestFeatureLabel][value] = self.CreateTree(splited_data, sub_labels)
        #print(treeDict)
        return treeDict
 
    def fit(self):
        self.tree = self.CreateTree(self.data, self.labels)
 
    def predict_vec(self, vec, input_tree=None):
        if input_tree==None:
            input_tree = self.tree

        featureIndex = self.labels.index(list(input_tree.keys())[0])
        #print(list(input_tree.keys()),featureIndex)
        secTree = list(input_tree.values())[0]
        #print(list(input_tree.values())[0])
        # #print(vec[featureIndex])
        vec_feature_val = vec[featureIndex]
        #print(vec_feature_val, secTree, secTree.get(vec_feature_val))

        if type(secTree.get(vec_feature_val)) != dict:
            #print(secTree.get(vec_feature_val))
            return secTree.get(vec_feature_val)
        else:
            #print(secTree.get(vec_feature_val))
            return self.predict_vec(vec, secTree.get(vec_feature_val))
 
    def predict(self, x):
        out_put=[]
        for i in x:
            out_put.append(self.predict_vec(i))
        return out_put
 
 
if __name__ == '__main__':
    x = np.array([[1, 1],[1, 1],[1, 0],[0, 1],[0, 1]])
    y = np.array([[1], [1], [0], [0], [0]])
    labels = [0, 1]
    id3 = ID3(x, y, labels)
    id3.fit()
    #print(id3.tree)
    print('预测值为：', id3.predict([[1, 0]]))

 
