#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

/**
* @description: 	KNN模型
*/
class KNN
{
public:
	/**
	* @description: 	构造函数
	* @param x			特征
	* @param y			标签
	* @param k			邻近数
	* @param p			度量方式
	*/
	KNN(std::vector<std::vector<float>> x, std::vector<float> y, int k, float p) : m_x(x), m_y(y), m_k(k), m_p(p) { ; };

	/**
	* @description: 	预测
	* @param x			输入特征
	* @return
	*/
	int predict(std::vector<std::vector<float>> x)
	{
		x.resize(m_x.size());
		for (size_t i = 0; i < x.size(); i++)
		{
			x[i] = x[0];
		}

		std::vector<std::vector<float>> diff = x;
		for (size_t i = 0; i < diff.size(); i++)
		{
			for (size_t j = 0; j < diff[0].size(); j++)
			{
				diff[i][j] -= m_x[i][j];
			}
		}

		std::vector<float> dist(diff.size(), 0);
		for (size_t i = 0; i < diff.size(); i++)
		{
			for (size_t j = 0; j < diff[0].size(); j++)
			{
				dist[i] += pow(diff[i][j], m_p);
			}
			dist[i] = pow(dist[i], 1.0 / m_p);
		}

		std::vector<int> dist_sorted(dist.size());
		for (size_t i = 0; i != dist_sorted.size(); ++i)
			dist_sorted[i] = i;
		std::sort(dist_sorted.begin(), dist_sorted.end(), [&dist](size_t i, size_t j) { return dist[i] < dist[j]; });

		std::map<float, int> count;
		for (size_t i = 0; i < m_k; i++)
		{
			float vote = m_y[dist_sorted[i]];
			count[vote] += 1;
		}

		return count.rbegin()->first;
	}

private:
	/**
	* @description: 	特征
	*/
	std::vector<std::vector<float>> m_x;

	/**
	* @description: 	标签
	*/
	std::vector<float> m_y;

	/**
	* @description: 	邻近数
	*/
	int m_k;

	/**
	* @description: 	度量方式
	*/
	float m_p;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{0, 10}, {1, 8}, {10, 1}, {7, 4}};
	std::vector<float> y = {0, 0, 1, 1};
	KNN knn = KNN(x, y, 3, 2);
	std::cout << "预测值为：" << knn.predict({{6, 2}}) << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}
