import numpy as np

class AdaBoost:
    #单层决策树生成函数，通过阀值比较对数据进行分类，在阀值一边的数据分到类别-1，而在另一边的数据分到类别+1
    def stumpClassify(self, dataMatrix, dimen, threshVal, threshIneq):  #数据集，数据集列数，阈值，比较方式：lt，gt
        retArray = np.ones((np.shape(dataMatrix)[0], 1))  #将数组的全部元素设置为1
        #print('d',dataMatrix,dimen,threshVal,threshIneq,retArray)
        #lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到retArray
        if threshIneq == 'lt':
            retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:, dimen] > threshVal] = -1.0
        #print(retArray)
        return retArray  #返回分类结果

    #遍历stumpClassify()函数所有的可能输入值，并找到数据集上的最佳的单层决策树
    def buildStump(self, dataArr, classLabels, D):  #数据集，数据标签，权重向量
        dataMatrix = np.mat(dataArr)  #使输入数据符合矩阵格式
        labelMat = np.mat(classLabels).T  #标签，转置为列向量
        m, n = np.shape(dataMatrix)  #矩阵形状，m为样本个数，n为每个样本的特征个数
        #print(dataMatrix,m,n)
        numSteps = 10.0  #初始化步数，用于在特征的所有可能值上进行遍历
        bestStump = {}  #创建一个空字典，用于存储给定权重向量D时所得到的最佳决策树的相关信息
        bestClasEst = np.mat(np.zeros((m, 1)))  #初始化类别估计值
        minError = np.inf  #一开始初始化为正无穷大，之后用于寻找可能的最小错误率

        for i in range(n):  #遍历数据集的所有特征
            #print(dataMatrix[:, i])
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps  #通过计算最大值和最小值差值除以步数来确定步长
            for j in range(-1, int(numSteps) + 1):  #遍历每个步长
                for inequal in ['lt', 'gt']:  #遍历每个大于和小于不等式
                    threshVal = rangeMin + j * stepSize  #设定阀值
                    predictedVals = self.stumpClassify(dataMatrix, i, threshVal, inequal)  #调用stumpClassify函数，通过阀值比较对数据进行分类
                    errArr = np.mat(np.ones((m, 1)))  #错误列向量，将预测结果与真实类别比较，不相同则对应位置设为1
                    errArr[predictedVals == labelMat] = 0  #相同位置设为0
                    #print('e',errArr)
                    weightedError = D.T * errArr  #将错误向量和权值向量相乘并求和（横向量*列向量=对应元素相乘求和）
                    #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                    #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                    if weightedError < minError:
                        minError = weightedError
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, bestClasEst  #返回最佳单层决策树，最小错误率，类别估计值

    #完整AdaBoost算法实现
    def fit(self, x, y, iters=10):  #数据集，类别标签，迭代次数
        self.weakClassArr = []  #建立一个空列表，用于存储单层决策树的信息
        m = x.shape[0]  #数据集行数
        D = np.mat(np.ones((m, 1)) / m)  #初始化向量D每个值均为1/m，D包含每个数据点的权重，在后续的迭代中会增加错误预测的权重，相应的减少正确预测的权重
        aggClassEst = np.mat(np.zeros((m, 1)))  #初始化列向量，记录每个数据点的类别估计累计值
        for i in range(iters):  #遍历迭代次数，直到达到遍历次数或者错误率为0停止迭代
            bestStump, error, classEst = self.buildStump(x, y, D)  #调用buildStump构建一个单层决策树
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  #根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
            bestStump['alpha'] = alpha  #alpha加入到字典中
            #print("alph'",alpha)
            self.weakClassArr.append(bestStump)  #再添加到列表
            #print("classLabels",classLabels,"classEst:", classEst)

            #计算下一次迭代中的新权重向量D
            expon = np.multiply(-alpha * np.mat(y).T, classEst)
            #print(expon)
            D = np.multiply(D, np.exp(expon))
            D = D / D.sum()
            #print("D:", D)

            aggClassEst += alpha * classEst  #累加类别估计值，该值为浮点型
            #print("aggClassEst:", aggClassEst)
            #print(np.sign(aggClassEst) != np.mat(classLabels).T)
            aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(y).T, np.ones((m, 1)))  #通过sign()函数得到二值分类结果，与真实类别标签比较，计算错误分类的个数
            errorRate = aggErrors.sum() / m  #计算错误率
            #print("total error: ", errorRate)  #打印每次迭代的错误率

            if errorRate == 0.0:  #如果某次迭代之后的错误率为0，就退出迭代过程，则不需要达到预先设定的迭代次数
                break
        #print(self.weakClassArr)
        return self.weakClassArr

    def predict(self, x):  #待分类样本，多个弱分类器组成的数组
        dataMatrix = np.mat(x)  #将待分类样本转为矩阵
        m = dataMatrix.shape[0]  #得到测试样本的个数
        aggClassEst = np.mat(np.zeros((m, 1)))  #构建一个0列向量，作用同上
        for i in range(len(self.weakClassArr)):  #遍历所有弱分类器     
            classEst = self.stumpClassify(dataMatrix, self.weakClassArr[i]['dim'], self.weakClassArr[i]['thresh'], self.weakClassArr[i]['ineq'])  #基于stumpClassify()对每个弱分类器得到一个类别的估计值
            aggClassEst += self.weakClassArr[i]['alpha'] * classEst  #输出的类别值乘以该单层决策树的alpha权重再累加到aggClassEst上
            #print('aggClassEst',aggClassEst)  #打印结果
        return np.sign(aggClassEst)  #返回分类结果，aggClassEst大于0则返回+1，小于0则返回-1


if __name__ == '__main__':
    x = np.array([[1., 2.1],[2., 1.1],[1.3, 1.],[1., 1.],[2., 1.]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    ada = AdaBoost()

    D = np.mat(np.ones((5, 1)) / 5)   #初始化权重向量
    print('最佳单层决策树相关信息：', ada.buildStump(x, y, D))

    ada.fit(x, y, iters=100)
    print('预测值为：', ada.predict([[0, 0]]))

    