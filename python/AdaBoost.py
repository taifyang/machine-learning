import numpy as np


class AdaBoost:
    '''
    description:    AdaBoost模型
    '''

    def stumpClassify(self, dataMatrix, dimen, threshVal, threshIneq):
        '''
        description:        单层决策树生成函数，通过阀值比较对数据进行分类，在阀值一边的数据分到类别-1，而在另一边的数据分到类别+1
        param self
        param dataMatrix    数据集
        param dimen         数据集列数
        param threshVal     阈值
        param threshIneq    比较方式
        return              分类结果
        '''
        retArray = np.ones((np.shape(dataMatrix)[0], 1))
        if threshIneq == 'lt':
            retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
        else:
            retArray[dataMatrix[:, dimen] > threshVal] = -1.0
        return retArray

    def buildStump(self, dataArr, classLabels, D):
        '''
        description:        遍历stumpClassify()函数所有的可能输入值，并找到数据集上的最佳的单层决策树
        param self
        param dataArr       数据集
        param classLabels   数据标签
        param D             权重向量
        return              存储给定权重向量D时所得到的最佳决策树的相关信息，最小错误率，类别估计值
        '''
        dataMatrix = np.mat(dataArr)
        labelMat = np.mat(classLabels).T
        m, n = np.shape(dataMatrix)

        numSteps = 10.0
        bestStump = {}
        classEst = np.mat(np.zeros((m, 1)))
        minError = np.inf

        for i in range(n):
            rangeMin = dataMatrix[:, i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize = (rangeMax - rangeMin) / numSteps
            for j in range(-1, int(numSteps) + 1):
                for inequal in ['lt', 'gt']:
                    threshVal = rangeMin + j * stepSize
                    predictedVals = self.stumpClassify(
                        dataMatrix, i, threshVal, inequal)
                    errArr = np.mat(np.ones((m, 1)))
                    errArr[predictedVals == labelMat] = 0
                    weightedError = D.T * errArr

                    if weightedError < minError:
                        minError = weightedError
                        classEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        return bestStump, minError, classEst

    def fit(self, x, y, iters=10):
        '''
        description:    完整AdaBoost算法实现
        param self
        param x         特征
        param y         类别标签
        param iters     迭代次数
        return          存储单层决策树的信息
        '''
        self.weakClassArr = []
        m = x.shape[0]
        D = np.mat(np.ones((m, 1)) / m)
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(iters):
            bestStump, error, classEst = self.buildStump(x, y, D)
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
            bestStump['alpha'] = alpha
            self.weakClassArr.append(bestStump)

            expon = np.multiply(-alpha * np.mat(y).T, classEst)
            D = np.multiply(D, np.exp(expon))
            D = D / D.sum()

            aggClassEst += alpha * classEst
            aggErrors = np.multiply(
                np.sign(aggClassEst) != np.mat(y).T, np.ones((m, 1)))
            errorRate = aggErrors.sum() / m

            if errorRate == 0.0:
                break
        return self.weakClassArr

    def predict(self, x):
        '''
        description:    预测
        param self
        param x         待分类样本
        return          分类结果
        '''
        dataMatrix = np.mat(x)
        m = dataMatrix.shape[0]
        aggClassEst = np.mat(np.zeros((m, 1)))
        for i in range(len(self.weakClassArr)):
            # 基于stumpClassify()对每个弱分类器得到一个类别的估计值
            classEst = self.stumpClassify(
                dataMatrix, self.weakClassArr[i]['dim'], self.weakClassArr[i]['thresh'], self.weakClassArr[i]['ineq'])
            aggClassEst += self.weakClassArr[i]['alpha'] * classEst
        return np.sign(aggClassEst)


if __name__ == '__main__':
    x = np.array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    ada = AdaBoost()
    D = np.mat(np.ones((5, 1)) / 5)
    ada.buildStump(x, y, D)
    ada.fit(x, y, iters=100)
    print('预测值为：', ada.predict([[0, 0]]))
