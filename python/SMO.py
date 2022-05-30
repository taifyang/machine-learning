import numpy as np
import random

class SimpleSMO(object):  
    def __init__(self,x,y,b,c,tolerance,max_iter):      
        self.x = x
        self.y = y
        self.b = b
        self.c = c
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.alpha = np.zeros((self.x.shape[0],1))
               
    def g(self,x_i):
        #计算对输入x_i的预测值, 此处必须reshape(-1,1)
        #print((np.dot(self.x,x_i.T).reshape(-1,1))) 
        return np.sum(self.alpha * self.y.reshape(-1, 1) * (np.dot(self.x,x_i.T).reshape(-1,1))) + self.b

    def Error(self,x_i,y_i):
        #计算预测值与输入值的误差 
        #print(self.g(x_i) - y_i)
        return self.g(x_i) - y_i
    
    def SelectJ(self,i):
        #简化版SMO:随机选择第二个优化变量j，并使其不等于第一个i
        j = i
        while (j==i):
            j = int(random.uniform(0,self.x.shape[0]))
        return j
    
    def Kernal(self,m,n):
        #定义核函数，用于计算Kij，本例中Kij = x[i].*x[j]
        return self.x[m].dot(self.x[n].T)       
    
    def Optimization(self):
        iter = 0 
        #while循环用于判定变量是否继续更新，iter只有在alpha不再发生变化时才会更新
        while (iter < self.max_iter):
            #alphaPairsChanged用于建立alpha是否改变的标志
            alphaPairsChanged = 0
            #建立for循环，for循环作为外层循环，寻找一个变量
            for i in range(self.alpha.size):
                #获得基于当前alpha下的第i个样本的误差
                E_i = self.Error(self.x[i],self.y[i])  
                #print(E_i)              
                ##选择第一个变量的要求：alpha_i是否严重违反kkt条件
                if (y[i] * E_i < -self.tolerance and self.alpha[i] < self.c ) or (y[i] * E_i > self.tolerance and self.alpha[i] > 0 ):
                    #违反kkt条件成立，随机选择第二个优化变量aplha_j（简化版SMO算法）
                    j = self.SelectJ(i)
                    #获得基于当前alpha下的第j个样本的误差
                    E_j = self.Error(self.x[j],self.y[j])
                    #记录未更新前alpha_i,alpha_j的值（即alpha_old值）为计算new值作准备
                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()
                    #根据alpha_i_old，alpha_j_old的值获得alpha_j_new的取值范围
                    if (self.y[i] != self.y[j]):
                        L = max(0,alpha_j_old-alpha_i_old)
                        H = min(self.c,self.c + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0,alpha_j_old + alpha_i_old - self.c)
                        H = min(self.c,alpha_j_old + alpha_i_old)
                    #print(L,H)
                    if L == H:
                        print("L=H")
                        continue
                    #计算eta
                    eta = 2 * self.Kernal(i,j)-self.Kernal(i,i)-self.Kernal(j,j)
                    if eta>= 0:
                        print("eta>=0")
                        continue
                    #根据alpha_j_old,eta，y_i,E_i,E_j更新alpha_j_new_unc未剪辑的更新值
                    alpha_j_new_unc = alpha_j_old - y[j]*(E_i-E_j)/eta
                    #获得剪辑后的更新值并保存
                    self.alpha[j] = np.clip(alpha_j_new_unc,L,H)
                    ##选择第二个变量的要求：alpha_j具有足够大的变化
                    if abs(self.alpha[j]-alpha_j_old) < 0.00001:
                        print("j not moving enough")
                        continue
                    #根据alpha_j_old 和 更新后的self.alpha[j] 更新 self.alpha[i]
                    self.alpha[i] += self.y[i]*self.y[j]*(alpha_j_old-self.alpha[j])
                    #更新常数项b_i_new
                    b_i_new = self.b - E_i -y[i]*self.Kernal(i,i)*(self.alpha[i]-alpha_i_old) - y[j]*self.Kernal(j,i)*(self.alpha[j]-alpha_j_old)
                    #更新常数项b_j_new
                    b_j_new = self.b - E_j -y[i]*self.Kernal(i,j)*(self.alpha[i]-alpha_i_old) - y[j]*self.Kernal(j,j)*(self.alpha[j]-alpha_j_old)
                    if (self.alpha[i]>0 and self.alpha[i]<self.c):
                        self.b = b_i_new
                    elif (self.alpha[j]>0 and self.alpha[j]<self.c):
                        self.b = b_j_new                    
                    else:
                        self.b = (b_i_new + b_j_new)/2
                    #若程序无中断，alpha必然发生改变，所以标志也要变化
                    alphaPairsChanged += 1
                    print("External loop: %d; Internal loop i :%d; alphaPairsChanged :%d" % (iter,i,alphaPairsChanged))
            #只有alpha不再改变时（此时意味着很有可能是最优解），迭代次数iter更新从而验证是否为最优解
            if (alphaPairsChanged == 0):
                iter += 1
            #alpha改变时，迭代次数iter置0
            else:
                iter = 0
            print("Iteration number : %d" % iter)    


if __name__ == '__main__':
    x = np.array([[4,2], [3,3], [8,-2], [2,-4], [8,1]])
    y = np.array([-1,-1,1,-1,1])
    smo = SimpleSMO(x,y,0,0.6,0.001,10)
    #print(smo.Error(x[0],y[0]))
    smo.Optimization()

