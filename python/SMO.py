import numpy as np
import random


class SimpleSMO(object):
    '''
    description:    SimpleSMO模型
    '''

    def __init__(self, x, y, b, c, tolerance, max_iter):
        '''
        description:    构造方法
        param self
        param x         特征
        param y         标签
        param b         常数项
        param c         范围约束
        param tolerance 容忍度
        param max_iter  最大迭代次数
        '''
        self.x = x
        self.y = y
        self.b = b
        self.c = c
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.alpha = np.zeros((self.x.shape[0], 1))

    def g(self, x_i):
        '''
        description:    计算对输入x_i的预测值
        param self
        param x_i       输入特征
        return          预测值
        '''
        return np.sum(self.alpha * self.y.reshape(-1, 1) * (np.dot(self.x, x_i.T).reshape(-1, 1))) + self.b

    def Error(self, x_i, y_i):
        '''
        description:    计算预测值与输入值的误差
        param self
        param x_i       输入特征
        param y_i       输入标签
        return          预测值与输入值的误差
        '''
        return self.g(x_i) - y_i

    def SelectJ(self, i):
        '''
        description:    随机选择第二个优化变量j，并使其不等于第一个i
        param self
        param i         索引i
        return          第二个优化变量j
        '''
        j = i
        while (j == i):
            j = int(random.uniform(0, self.x.shape[0]))
        return j

    def Kernal(self, m, n):
        '''
        description:    核函数，用于计算Kij，本例中Kij = x[i].*x[j]
        param self
        param m         索引m
        param n         索引n
        return          Kij
        '''
        return self.x[m].dot(self.x[n].T)

    def Optimization(self):
        '''
        description:    优化
        param self
        '''
        iter = 0
        while (iter < self.max_iter):
            alphaPairsChanged = 0
            for i in range(self.alpha.size):
                E_i = self.Error(self.x[i], self.y[i])
                if (y[i] * E_i < -self.tolerance and self.alpha[i] < self.c) or (y[i] * E_i > self.tolerance and self.alpha[i] > 0):
                    j = self.SelectJ(i)
                    E_j = self.Error(self.x[j], self.y[j])
                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()
                    if (self.y[i] != self.y[j]):
                        L = max(0, alpha_j_old-alpha_i_old)
                        H = min(self.c, self.c + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_j_old + alpha_i_old - self.c)
                        H = min(self.c, alpha_j_old + alpha_i_old)
                    if L == H:
                        print("L=H")
                        continue
                    eta = 2 * self.Kernal(i, j) - self.Kernal(i, i)-self.Kernal(j, j)
                    if eta >= 0:
                        print("eta>=0")
                        continue
                    alpha_j_new_unc = alpha_j_old - y[j]*(E_i-E_j)/eta
                    self.alpha[j] = np.clip(alpha_j_new_unc, L, H)
                    if abs(self.alpha[j]-alpha_j_old) < 0.00001:
                        print("j not moving enough")
                        continue
                    self.alpha[i] += self.y[i]*self.y[j] * (alpha_j_old-self.alpha[j])
                    b_i_new = self.b - E_i - y[i]*self.Kernal(i, i)*(self.alpha[i]-alpha_i_old) - y[j]*self.Kernal(j, i)*(self.alpha[j]-alpha_j_old)
                    b_j_new = self.b - E_j - y[i]*self.Kernal(i, j)*(self.alpha[i]-alpha_i_old) - y[j]*self.Kernal(j, j)*(self.alpha[j]-alpha_j_old)
                    if (self.alpha[i] > 0 and self.alpha[i] < self.c):
                        self.b = b_i_new
                    elif (self.alpha[j] > 0 and self.alpha[j] < self.c):
                        self.b = b_j_new
                    else:
                        self.b = (b_i_new + b_j_new)/2
                    alphaPairsChanged += 1
                    print("External loop: %d; Internal loop i :%d; alphaPairsChanged :%d" % (iter, i, alphaPairsChanged))
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print("Iteration number : %d" % iter)


if __name__ == '__main__':
    x = np.array([[4, 2], [3, 3], [8, -2], [2, -4], [8, 1]])
    y = np.array([-1, -1, 1, -1, 1])
    smo = SimpleSMO(x, y, 0, 0.6, 0.001, 10)
    smo.Optimization()
