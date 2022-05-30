import numpy as np
import pandas as pd

class PCA(object):
    """定义PCA类"""
    def __init__(self, x, n_components=None):
        """x的数据结构应为ndarray"""
        self.x = x
        self.dimension = x.shape[1]       
        if n_components and n_components >= self.dimension:
            print("n_components error")        
        self.n_components = n_components      
    
    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_T = np.transpose(self.x)                           #矩阵转秩
        x_cov = np.cov(x_T)                                  #协方差矩阵
        a, b = np.linalg.eig(x_cov)                          #a为特征值，b为特征向量
        m = a.shape[0]
        c = np.hstack((a.reshape((m,1)), b))
        #print(a,b,c)
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort_values(0, ascending=False)    #按照特征值大小降序排列特征向量
        return c_df_sort
        
    def explained_variance_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]       
              
    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        variance = self.explained_variance_()        
        #print(c_df_sort,variance)
        if self.n_components:                                #指定降维维度
            p = c_df_sort.values[0:self.n_components, 1:]
            y = np.dot(p, np.transpose(self.x))              #矩阵叉乘
            return np.transpose(y)      
        variance_sum = sum(variance)                         #利用方差贡献度自动选择降维维度
        variance_radio = variance / variance_sum       
        variance_contribution = 0
        for R in range(self.dimension):
            variance_contribution += variance_radio[R]       #前R个方差贡献度之和
            if variance_contribution >= 0.99:
                break            
        p = c_df_sort.values[0:R+1, 1:]                      #取前R个特征向量
        y = np.dot(p, np.transpose(self.x))                  #矩阵叉乘
        return np.transpose(y)
  

if __name__ == '__main__':
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(x)
    print(x)
    print(pca.reduce_dimension())
    print(pca.explained_variance_())
