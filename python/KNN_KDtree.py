import numpy as np
import random
from copy import deepcopy
from numpy.linalg import norm
from collections import Counter


def partition_sort(arr, k, key=lambda x: x):
    '''
    description:    以枢纽(位置k)为中心将数组划分为两部分, 枢纽左侧的元素不大于枢纽右侧的元素
    param arr       待划分数组
    param k         枢纽前部元素个数
    param key       比较方式
    '''
    start, end = 0, len(arr) - 1
    assert 0 <= k <= end
    while True:
        i, j, pivot = start, end, deepcopy(arr[start])
        while i < j:
            while i < j and key(pivot) <= key(arr[j]):
                j -= 1
            if i == j: break
            arr[i] = arr[j]
            i += 1
            while i < j and key(arr[i]) <= key(pivot):
                i += 1
            if i == j: break
            arr[j] = arr[i]
            j -= 1
        arr[i] = pivot

        if i == k:
            return
        elif i < k:
            start = i + 1
        else:
            end = i - 1


def max_heapreplace(heap, new_node, key=lambda x: x[1]):
    '''
    description:    大根堆替换堆顶元素
    param heap      大根堆/列表
    param new_node  新节点
    param key
    '''
    heap[0] = new_node
    root, child = 0, 1
    end = len(heap) - 1
    while child <= end:
        if child < end and key(heap[child]) < key(heap[child + 1]):
            child += 1
        if key(heap[child]) <= key(new_node):
            break
        heap[root] = heap[child]
        root, child = child, 2 * child + 1
    heap[root] = new_node


def max_heappush(heap, new_node, key=lambda x: x[1]):
    '''
    description:    大根堆插入元素
    param heap      大根堆/列表
    param new_node  新节点
    param key
    '''
    heap.append(new_node)
    pos = len(heap) - 1
    while 0 < pos:
        parent_pos = pos - 1 >> 1
        if key(new_node) <= key(heap[parent_pos]):
            break
        heap[pos] = heap[parent_pos]
        pos = parent_pos
    heap[pos] = new_node


class KDNode:
    '''
    description:    kd树节点
    '''

    def __init__(self, data=None, label=None, left=None, right=None, axis=None, parent=None):
         '''
        description:    构造方法
        param self
        param data      数据
        param label     数据标签
        param left      左孩子节点
        param right     右孩子节点
        param axis      分割轴
        param parent    父节点
        '''
        self.data = data
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis
        self.parent = parent

class KDTree:
    '''
    description:    kd树
    '''

    def __init__(self, x, y=None):
        '''
        description:    构造方法
        param self
        param x         输入特征
        param y         输入标签
        '''
        self.root = None
        self.y_valid = False if y is None else True
        self.create(x, y)

    def create(self, x, y=None):
        '''
        description:    构建kd树
        param self
        param x         输入特征
        param y         输入标签
        '''
       
        def create_(x, axis, parent=None):
            '''
            description:    递归生成kd树
            param x         合并标签后输入集
            param axis      切分轴
            param parent    父节点
            return          KDNode
            ''' 
            n_samples = np.shape(x)[0]
            if n_samples == 0:
                return None
            mid = n_samples >> 1            
            partition_sort(x, mid, key=lambda x: x[axis])

            if self.y_valid:
                kd_node = KDNode(x[mid][:-1], x[mid][-1], axis=axis, parent=parent)
            else:
                kd_node = KDNode(x[mid], axis=axis, parent=parent)
            
            next_axis = (axis + 1) % k_dimensions
            kd_node.left = create_(x[:mid], next_axis, kd_node)
            kd_node.right = create_(x[mid + 1:], next_axis, kd_node)
            return kd_node

        k_dimensions = np.shape(x)[1]
        if y is not None:
            x = np.hstack((np.array(x), np.array([y]).T)).tolist()
        self.root = create_(x, 0)

    def search_knn(self, point, k, p=2):
        '''
        description:        kd树中搜索k个最近邻样本
        param self
        param point         样本点
        param k             近邻数
        param p             度量方式
        return              k个最近邻样本
        '''

        def search_knn_(kd_node):
            '''
            description:    搜索k近邻节点
            param kd_node   KDNode
            '''
            if kd_node is None:
                return
            data = kd_node.data
            distance = np.linalg.norm(np.array(data) - np.array(point), ord=p, axis=None, keepdims=False) #计算范数
            if len(heap) < k:
                max_heappush(heap, (kd_node, distance))
            elif distance < heap[0][1]:
                max_heapreplace(heap, (kd_node, distance))

            axis = kd_node.axis
            if abs(point[axis] - data[axis]) < heap[0][1] or len(heap) < k:
                search_knn_(kd_node.left)
                search_knn_(kd_node.right)
            elif point[axis] < data[axis]:
                search_knn_(kd_node.left)
            else:
                search_knn_(kd_node.right)

        if self.root is None:
            raise Exception('kd-tree must be not null.')
        if k < 1:
            raise ValueError("k must be greater than 0.")
       
        heap = []
        search_knn_(self.root)
        return sorted(heap, key=lambda x: x[1])

class KNeighborsClassifier:
    '''
    description:    K近邻分类器
    '''
  
    def __init__(self, k, p=2):
        '''
        description:    构造方法
        param self
        param k         近邻数
        param p         度量方式
        '''  
        self.k = k
        self.dist = p
        self.kd_tree = None

    def fit(self, x, y):
        '''
        description:    建立kd树
        param self
        param x         特征
        param y         标签
        '''
        self.kd_tree = KDTree(x, y)

    def predict(self, x):
        '''
        description:    预测类别
        param self
        param x         特征
        return          预测值
        '''
        if self.kd_tree is None:
            raise TypeError('Classifier must be fitted before predict!')
        search_knn = lambda x: self.kd_tree.search_knn(point=x, k=self.k, p=self.dist)
        y_pre = []
        for x in x:
            y = Counter(r[0].label for r in search_knn(x)).most_common(1)[0][0]
            y_pre.append(y)
        return y_pre


if __name__ == '__main__':
    x=np.array([[0, 10], [1, 8], [10, 1], [7, 4]])
    y=np.array([0, 0, 1, 1])
    knn = KNeighborsClassifier(3)
    knn.fit(x, y)
    print("预测值为：",knn.predict(np.array([[6, 2]])))

