#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <assert.h>

void partition_sort(std::vector<std::vector<float>>& arr, int k, int key)
{
	int start = 0, end = arr.size() - 1;
	assert(k >= 0 && k <= end);
	while (true)
	{
		int i = start, j = end;
		std::vector<float> pivot = arr[start];
		while (i < j)
		{
			while (i<j && pivot[key] <= arr[j][key])  j -= 1;
			if (i == j)    break;
			arr[i] = arr[j];
			i += 1;
			while (i<j && arr[i][key] <= pivot[key])  i += 1;
			if (i == j)    break;
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

class KDNode
{
public:
	KDNode(std::vector<float> data = {}, float label = -1, int axis = -1, KDNode* left = NULL, KDNode* right = NULL, KDNode* parent = NULL) :
		m_data(data), m_label(label), m_axis(axis), m_left(left), m_right(right), m_parent(parent){};

public:
	std::vector<float> m_data;
	float m_label;
	int m_axis;
	KDNode* m_left;
	KDNode* m_right;
	KDNode* m_parent;
};


struct Tuple
{
	KDNode* kd_node;
	float distance;
};

void max_heapreplace(std::vector<Tuple>& heap, Tuple new_node)
{
	heap[0] = new_node;
	int root = 0, child = 1;
	int end = heap.size() - 1;
	while (child <= end)
	{
		if (child<end && heap[child].distance<heap[child + 1].distance)
			child += 1;
		if (heap[child].distance < new_node.distance)
			break;
		heap[root] = heap[child];
		root = child;
		child = 2 * child + 1;
	}
	heap[root] = new_node;
}

void max_heappush(std::vector<Tuple>& heap, Tuple new_node)
{
	heap.push_back(new_node);
	int pos = heap.size() - 1;
	while (0<pos)
	{
		int parent_pos = (pos - 1) >> 1;
		if (new_node.distance <= heap[parent_pos].distance)
				break;
		heap[pos] = heap[parent_pos];
		pos = parent_pos;
	}
	heap[pos] = new_node;
}

class KDTree
{
public:
	KDTree(std::vector<std::vector<float>> X, std::vector<float> y = {})
	{
		m_root = NULL;
		if (y.size())	m_y_valid = true;
		else	m_y_valid = false;
		create(X, y);
	}

	KDNode* create_(std::vector<std::vector<float>> X, int axis, int k_dimensions, KDNode* parent = NULL)
	{
		int n_samples = X.size();
		if (n_samples == 0)
			return NULL;
		int mid = n_samples >> 1;

		partition_sort(X, mid, axis);

		KDNode* kd_node;
		if (m_y_valid)
		{
			std::vector<float> X_data(X[0].size() - 1);
			for (size_t i = 0; i < X_data.size(); i++)
			{
				X_data[i] = X[mid][i];
			}
			kd_node = new KDNode(X_data, X[mid][X[0].size()-1], axis, parent);
		}
		else {
			kd_node = new KDNode(X[mid], -1, axis, parent);
		}
		//std::cout<<"data:" << kd_node->m_data[0] <<" " << kd_node->m_data[1] << std::endl;

		int next_axis = (axis + 1) % k_dimensions;
		std::vector<std::vector<float>> Xleft_data(mid, std::vector<float>(X[0].size()));
		for (size_t i = 0; i < Xleft_data.size(); i++)
		{
			for (size_t j = 0; j < Xleft_data[0].size(); j++)
			{
				Xleft_data[i][j] = X[i][j];
			}
		}		
		kd_node->m_left = create_(Xleft_data, next_axis, k_dimensions, kd_node);

		std::vector<std::vector<float>> Xright_data(X.size() - mid - 1, std::vector<float>(X[0].size()));
		for (size_t i = 0; i < Xright_data.size(); i++)
		{
			for (size_t j = 0; j < Xright_data[0].size(); j++)
			{
				Xright_data[i][j] = X[i + mid + 1][j];
			}
		}
		kd_node->m_right = create_(Xright_data, next_axis, k_dimensions, kd_node);

		return kd_node;
	}

	void create(std::vector<std::vector<float>> X, std::vector<float> y = {})
	{
		std::cout << "building kd-tree..." << std::endl;
		int k_dimensions = X[0].size();
		if (y.size())
		{
			for (size_t i = 0; i < X.size(); i++)
			{
				X[i].push_back(y[i]);
			}
		}
		m_root = create_(X, 0, k_dimensions);
	}

	float p_dist(std::vector<float> data, std::vector<float> point, float p)
	{
		float p_dist = 0.0;
		for (size_t i = 0; i < data.size(); i++)
		{
			p_dist += pow(data[i] - point[i], p);
		}
		return pow(p_dist, 1.0 / p);
	}

	void search_knn_(std::vector<Tuple>& heap, KDNode* kd_node, std::vector<float> point,int k, float p)
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
		else if(distance<heap[0].distance)
			max_heapreplace(heap, tuple);

		int axis = kd_node->m_axis;
		if (fabs(point[axis] - data[axis]) < heap[0].distance || heap.size() < k)
		{
			search_knn_(heap, kd_node->m_left, point, k, p);
			search_knn_(heap, kd_node->m_right, point, k, p);
		}
		else if(point[axis] < data[axis])
			search_knn_(heap, kd_node->m_left, point, k, p);
		else
			search_knn_(heap, kd_node->m_right, point, k, p);
	}

	std::vector<Tuple> search_knn(std::vector<float> point, int k, float p = 2)
	{
		if(m_root == NULL)
			throw std::exception("kd-tree must be not null.");
		if(k < 1)
			throw std::exception("k must be greater than 0.");

		std::vector<Tuple> heap = {};

		search_knn_(heap, m_root, point, k, p);
		//std::cout << heap.size()<< std::endl;

		std::sort(heap.begin(), heap.end(), [](Tuple tuple1, Tuple tuple2) {return tuple1.distance < tuple2.distance; });

		return heap;
	}	

private:
	KDNode* m_root;
	bool m_y_valid;
};

class KNeighborsClassifier
{
public:
	KNeighborsClassifier(int k, float p=2, KDTree* kd_tree=NULL):m_k(k), m_p(p), m_kd_tree(kd_tree) {};

	void fit(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		m_kd_tree = new KDTree(x, y);
	}

	std::vector<float> predict(std::vector<std::vector<float>> x)
	{
		if(m_kd_tree==NULL)
			throw std::exception("Classifier must be fitted before predict!");

		std::vector<float> y_pre(x.size());
		for (size_t i = 0; i < x.size(); i++)
		{
			std::map<float, int> count;
			std::vector<Tuple> heap = m_kd_tree->search_knn(x[i], m_k, m_p);		
			for (size_t i = 0; i < heap.size(); i++)
			{
				//std::cout << heap[i].kd_node->m_label << std::endl;
				++count[heap[i].kd_node->m_label];
			}

			std::vector<std::pair<float, int>>  count_vec;
			for (std::map<float, int>::iterator it = count.begin(); it != count.end(); it++)
			{
				count_vec.push_back(std::make_pair((*it).first, (*it).second));
			}
			std::sort(count_vec.begin(), count_vec.end(), [](std::pair<float, int> p1, std::pair<float, int> p2) {return p1.second < p2.second; });
			y_pre[i] = count_vec.rbegin()->first;
		}

		return y_pre;
	}

private:
	int m_k;
	float m_p;;
	KDTree* m_kd_tree;
};

int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 0, 10 },{ 1, 8 },{ 10, 1 },{ 7, 4 } };
	std::vector<float> y = { 0, 0, 1, 1 };

	//std::vector<std::vector<float>> arr = { {0.0f, 1.0f, 0.0f},{0.1f, 0.7777777777777778f, 0.0f}, {1.0, 0.0f, 1.0f}, {0.7f, 0.3333333333333333f, 1.0f} };
	//partition_sort(arr, 2, 0);
	//for (size_t i = 0; i < arr.size(); i++)
	//{
	//	for (size_t j = 0; j < arr[0].size(); j++)
	//	{
	//		std::cout << arr[i][j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	//KDTree kdtree = KDTree(x);
	//std::vector<Tuple> heap = kdtree.search_knn({ 6,2 }, 3);
	//for (size_t j = 0; j < heap.size(); j++)
	//{
	//	std::cout << heap[j].distance << std::endl;
	//}

	KNeighborsClassifier knn = KNeighborsClassifier(3);
	knn.fit(x, y);
	std::cout << "Ô¤²âÖµÎª£º" << knn.predict({ { 6,2 } })[0] << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}

