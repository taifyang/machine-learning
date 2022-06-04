import numpy as np
import pandas as pd


class PCA(object):
    '''
    description:    PCA模型
    '''

    def __init__(self, x, n_components=None):
        '''
        description:        构造函数
        param self
        param x             特征
        param n_components  降维维度
        '''
        self.x = x
        self.dimension = x.shape[1]
        if n_components and n_components >= self.dimension:
            raise ValueError("n_components error")
        self.n_components = n_components

    def get_feature(self):
        '''
        description:    求协方差矩阵C的特征值和特征向量
        param self
        return          按照特征值大小降序排列的特征向量
        '''
        x_T = np.transpose(self.x)
        x_cov = np.cov(x_T)
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m, 1)), b))
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort_values(0, ascending=False)
        return c_df_sort

    def explained_variance_(self):
        '''
        description:    计算方差值
        param self
        return          方差值
        '''
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]

    def reduce_dimension(self):
        '''
        description:    指定维度降维和根据方差贡献率自动降维
        param self
        return          降维结果
        '''
        c_df_sort = self.get_feature()
        variance = self.explained_variance_()
        if self.n_components:
            p = c_df_sort.values[0:self.n_components, 1:]
            y = np.dot(p, np.transpose(self.x))
            return np.transpose(y)
        variance_sum = sum(variance)
        variance_radio = variance / variance_sum
        variance_contribution = 0
        for R in range(self.dimension):
            variance_contribution += variance_radio[R]
            if variance_contribution >= 0.99:
                break
        p = c_df_sort.values[0:R+1, 1:]
        y = np.dot(p, np.transpose(self.x))
        return np.transpose(y)


if __name__ == '__main__':
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(x)
    print(x)
    print(pca.reduce_dimension())
    print(pca.explained_variance_())
