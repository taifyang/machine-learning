#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <assert.h>

/**
 * @description: 	以枢纽(位置k)为中心将数组划分为两部分, 枢纽左侧的元素不大于枢纽右侧的元素
 * @param arr		待划分数组
 * @param k			枢纽前部元素个数
 * @param key		比较方式
 */
void partition_sort(std::vector<std::vector<float>> &arr, int k, int key)
{
	int start = 0, end = arr.size() - 1;
	assert(k >= 0 && k <= end);
	while (true)
	{
		int i = start, j = end;
		std::vector<float> pivot = arr[start];
		while (i < j)
		{
			while (i < j && pivot[key] <= arr[j][key])
				j -= 1;
			if (i == j)
				break;
			arr[i] = arr[j];
			i += 1;
			while (i < j && arr[i][key] <= pivot[key])
				i += 1;
			if (i == j)
				break;
			arr[j] = arr[i];
			j -= 1;
		}
		arr[i] = pivot;

		if (i == k)
			return;
		else if (i < k)
			start = i + 1;
		else
			end = i - 1;
	}
}

/**
 * @description: 	kd树节点
 */
class KDNode
{
public:
	/**
         * @description: 	构造函数
         * @param data		数据
         * @param label		数据标签
         * @param axis		分割轴
         * @param left		左孩子节点
         * @param right		右孩子节点
         * @param parent	父节点
         */
	KDNode(std::vector<float> data = {}, float label = -1, int axis = -1, KDNode *left = NULL, KDNode *right = NULL, KDNode *parent = NULL) : m_data(data), m_label(label), m_axis(axis), m_left(left), m_right(right), m_parent(parent){};

public:
	/**
	 * @description: 	数据
	 */
	std::vector<float> m_data;

	/**
	 * @description: 	数据标签
	 */
	float m_label;

	/**
	 * @description: 	分割轴
	 */
	int m_axis;

	/**
	 * @description: 	左孩子节点
	 */
	KDNode *m_left;

	/**
	 * @description: 	右孩子节点
	 */
	KDNode *m_right;

	/**
	 * @description: 	父节点
	 */
	KDNode *m_parent;
};

/**
 * @description: 	元组
 */
struct Tuple
{
	/**
	 * @description: 	kd树节点
	 */
	KDNode *kd_node;

	/**
	 * @description: 	距离
	 */
	float distance;
};

/**
 * @description: 	大根堆替换堆顶元素
 * @param heap		大根堆/列表
 * @param new_node	新节点
 */
void max_heapreplace(std::vector<Tuple> &heap, Tuple new_node)
{
	heap[0] = new_node;
	int root = 0, child = 1;
	int end = heap.size() - 1;
	while (child <= end)
	{
		if (child < end && heap[child].distance < heap[child + 1].distance)
			child += 1;
		if (heap[child].distance < new_node.distance)
			break;
		heap[root] = heap[child];
		root = child;
		child = 2 * child + 1;
	}
	heap[root] = new_node;
}

/**
 * @description: 	大根堆插入元素
 * @param heap		大根堆/列表
 * @param new_node	新节点
 */
void max_heappush(std::vector<Tuple> &heap, Tuple new_node)
{
	heap.push_back(new_node);
	int pos = heap.size() - 1;
	while (0 < pos)
	{
		int parent_pos = (pos - 1) >> 1;
		if (new_node.distance <= heap[parent_pos].distance)
			break;
		heap[pos] = heap[parent_pos];
		pos = parent_pos;
	}
	heap[pos] = new_node;
}

/**
 * @description: 	kd树
 */
class KDTree
{
public:
	/**
     * @description: 	构造函数
     * @param x			输入特征
     * @param y			输入标签
     */
	KDTree(std::vector<std::vector<float>> x, std::vector<float> y = {})
	{
		m_root = NULL;
		if (y.size())
			m_y_valid = true;
		else
			m_y_valid = false;
		create(x, y);
	}

	/**
	 * @description: 		递归生成kd树
	 * @param x				合并标签后输入集
	 * @param axis			切分轴
	 * @param k_dimensions  维度
	 * @param parent		父节点
	 * @return 				KDNode
	 */
	KDNode *create_(std::vector<std::vector<float>> x, int axis, int k_dimensions, KDNode *parent = NULL)
	{
		int n_samples = x.size();
		if (n_samples == 0)
			return NULL;
		int mid = n_samples >> 1;

		partition_sort(x, mid, axis);

		KDNode *kd_node;
		if (m_y_valid)
		{
			std::vector<float> x_data(x[0].size() - 1);
			for (size_t i = 0; i < x_data.size(); i++)
			{
				x_data[i] = x[mid][i];
			}
			kd_node = new KDNode(x_data, x[mid][x[0].size() - 1], axis, parent);
		}
		else
		{
			kd_node = new KDNode(x[mid], -1, axis, parent);
		}

		int next_axis = (axis + 1) % k_dimensions;
		std::vector<std::vector<float>> xleft_data(mid, std::vector<float>(x[0].size()));
		for (size_t i = 0; i < xleft_data.size(); i++)
		{
			for (size_t j = 0; j < xleft_data[0].size(); j++)
			{
				xleft_data[i][j] = x[i][j];
			}
		}
		kd_node->m_left = create_(xleft_data, next_axis, k_dimensions, kd_node);

		std::vector<std::vector<float>> xright_data(x.size() - mid - 1, std::vector<float>(x[0].size()));
		for (size_t i = 0; i < xright_data.size(); i++)
		{
			for (size_t j = 0; j < xright_data[0].size(); j++)
			{
				xright_data[i][j] = x[i + mid + 1][j];
			}
		}
		kd_node->m_right = create_(xright_data, next_axis, k_dimensions, kd_node);

		return kd_node;
	}

	/**
     * @description: 	构建kd树
     * @param x			输入特征
     * @param y			输入标签
     */
	void create(std::vector<std::vector<float>> x, std::vector<float> y = {})
	{
		int k_dimensions = x[0].size();
		if (y.size())
		{
			for (size_t i = 0; i < x.size(); i++)
			{
				x[i].push_back(y[i]);
			}
		}
		m_root = create_(x, 0, k_dimensions);
	}

	/**
	 * @description: 	计算距离
	 * @param data		数据点
	 * @param point		样本点
	 * @param p			采用p范数度量
	 * @return 			距离值
	 */
	float p_dist(std::vector<float> data, std::vector<float> point, float p)
	{
		float p_dist = 0.0;
		for (size_t i = 0; i < data.size(); i++)
		{
			p_dist += pow(data[i] - point[i], p);
		}
		return pow(p_dist, 1.0 / p);
	}

	/**
     * @description: 	搜索k近邻节点
	 * @param heap		大根堆
     * @param kd_node	KDNode
     * @param point		样本点	
     * @param k			近邻数
     * @param p			度量方式
     */
	void search_knn_(std::vector<Tuple> &heap, KDNode *kd_node, std::vector<float> point, int k, float p)
	{
		if (kd_node == NULL)
			return;
		std::vector<float> data = kd_node->m_data;
		float distance = p_dist(data, point, p);

		Tuple tuple;
		tuple.kd_node = kd_node;
		tuple.distance = distance;
		if (heap.size() < k)
			max_heappush(heap, tuple);
		else if (distance < heap[0].distance)
			max_heapreplace(heap, tuple);

		int axis = kd_node->m_axis;
		if (fabs(point[axis] - data[axis]) < heap[0].distance || heap.size() < k)
		{
			search_knn_(heap, kd_node->m_left, point, k, p);
			search_knn_(heap, kd_node->m_right, point, k, p);
		}
		else if (point[axis] < data[axis])
			search_knn_(heap, kd_node->m_left, point, k, p);
		else
			search_knn_(heap, kd_node->m_right, point, k, p);
	}

	/**
	 * @description: 	kd树中搜索k个最近邻样本
	 * @param point		样本点
	 * @param k			近邻数
	 * @param p			度量方式
	 * @return			k个最近邻样本
	 */
	std::vector<Tuple> search_knn(std::vector<float> point, int k, float p = 2)
	{
		if (m_root == NULL)
			throw std::exception("kd-tree must be not null.");
		if (k < 1)
			throw std::exception("k must be greater than 0.");

		std::vector<Tuple> heap = {};
		search_knn_(heap, m_root, point, k, p);
		std::sort(heap.begin(), heap.end(), [](Tuple tuple1, Tuple tuple2) { return tuple1.distance < tuple2.distance; });
		return heap;
	}

private:
	/**
	 * @description: 	根节点
	 */
	KDNode *m_root;

	/**
	 * @description: 	标签真实值
	 */
	bool m_y_valid;
};

/**
 * @description: 	K近邻分类器
 */
class KNeighborsClassifier
{
public:
	/**
	 * @description: 	构造函数
     * @param k			近邻数
     * @param p			度量方式
	 * @param kd_tree   kd树
	 */
	KNeighborsClassifier(int k, float p = 2, KDTree *kd_tree = NULL) : m_k(k), m_p(p), m_kd_tree(kd_tree){};

	/**
	 * @description: 	建立kd树
	 * @param x			特征
	 * @param y			标签
	 */
	void fit(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		m_kd_tree = new KDTree(x, y);
	}

	/**
	 * @description: 	预测类别
	 * @param x			特征
	 * @return			预测值
	 */
	std::vector<float> predict(std::vector<std::vector<float>> x)
	{
		if (m_kd_tree == NULL)
			throw std::exception("Classifier must be fitted before predict!");

		std::vector<float> y_pre(x.size());
		for (size_t i = 0; i < x.size(); i++)
		{
			std::map<float, int> count;
			std::vector<Tuple> heap = m_kd_tree->search_knn(x[i], m_k, m_p);
			for (size_t i = 0; i < heap.size(); i++)
			{
				++count[heap[i].kd_node->m_label];
			}

			std::vector<std::pair<float, int>> count_vec;
			for (std::map<float, int>::iterator it = count.begin(); it != count.end(); it++)
			{
				count_vec.push_back(std::make_pair((*it).first, (*it).second));
			}
			std::sort(count_vec.begin(), count_vec.end(), [](std::pair<float, int> p1, std::pair<float, int> p2) { return p1.second < p2.second; });
			y_pre[i] = count_vec.rbegin()->first;
		}

		return y_pre;
	}

private:
	/**
	 * @description: 	近邻数
	 */
	int m_k;

	/**
     * @description: 	度量方式
     */
	float m_p;

	/**
	 * @description: 	kd树
	 */
	KDTree *m_kd_tree;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{0, 10}, {1, 8}, {10, 1}, {7, 4}};
	std::vector<float> y = {0, 0, 1, 1};
	KNeighborsClassifier knn = KNeighborsClassifier(3);
	knn.fit(x, y);
	std::cout << "预测值为：" << knn.predict({{6, 2}})[0] << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}
