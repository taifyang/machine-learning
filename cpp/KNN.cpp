#include <iostream>
#include <vector>
#include <map>
#include <algorithm>

//k�ڽ�ģ��
class KNN
{
public:
	KNN(std::vector<std::vector<float>> x, std::vector<float> y, int k, float p) : m_x(x), m_y(y), m_k(k), m_p(p) {};

	int predict(std::vector<std::vector<float>> x)
	{
		x.resize(m_x.size());
		for (size_t i = 0; i < x.size(); i++)
		{
			x[i] = x[0];
		}

		//����Ԥ�����ݺ�ѵ�����ݵĲ�ֵ
		std::vector<std::vector<float>> diff = x;
		for (size_t i = 0; i < diff.size(); i++)
		{
			for (size_t j = 0; j < diff[0].size(); j++)
			{
				diff[i][j] -= m_x[i][j];
			}
		}

		//���㷶��
		std::vector<float> dist(diff.size(), 0);
		for (size_t i = 0; i < diff.size(); i++)
		{
			for (size_t j = 0; j < diff[0].size(); j++)
			{
				dist[i] += pow(diff[i][j], m_p);
			}
			dist[i] = pow(dist[i], 1.0 / m_p);
		}

		//���ش�С�������������
		std::vector<int>  dist_sorted(dist.size());
		for (size_t i = 0; i != dist_sorted.size(); ++i) dist_sorted[i] = i;
		std::sort(dist_sorted.begin(), dist_sorted.end(), [&dist](size_t i, size_t j) {return dist[i] <  dist[j]; });

		//����ͶƱ
		std::map<float, int> count;
		for (size_t i = 0; i < m_k; i++)
		{
			float vote = m_y[dist_sorted[i]];
			count[vote] += 1;
		}

		//����ͶƱ��������ǩ
		return count.rbegin()->first;
	}
	
private:
	std::vector<std::vector<float>> m_x;
	std::vector<float> m_y;
	int m_k;
	float m_p;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 0, 10 },{ 1, 8 },{ 10, 1 },{ 7, 4 } };
	std::vector<float> y = { 0, 0, 1, 1 };

	KNN knn = KNN(x, y, 3, 2);
	std::cout << "Ԥ��ֵΪ��" << knn.predict({ {6,2} }) << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}
