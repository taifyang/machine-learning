#include <iostream>
#include <vector>
#include <time.h>

void printMat(std::vector<std::vector<float>> mat)
{
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat[0].size(); j++)
		{
			std::cout << mat[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

bool checkZeros(std::vector<std::vector<float>> mat)
{
	bool flag = true;
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat[0].size(); j++)
		{
			if (mat[i][j] != 0) flag = false;
		}
	}
	return flag;
}

std::vector<int> getminDistIndices(std::vector<std::vector<float>> clalist)
{
	std::vector<int> minDistIndices(clalist.size());
	for (size_t i = 0; i < clalist.size(); i++)
	{
		float minDist = INT_MAX;
		int minDistIndex = 0;
		for (size_t j = 0; j < clalist[0].size(); j++)
		{
			if (clalist[i][j] < minDist)
			{
				minDist = clalist[i][j];
				minDistIndex = j;
			}
		}
		//std::cout << minDistIndex << std::endl;
		minDistIndices[i] = minDistIndex;
	}
	return minDistIndices;
}

class KMeans
{
public:
	KMeans(std::vector<std::vector<float>> dataSet, int k) :m_dataSet(dataSet), m_k(k) {};

	std::vector<std::vector<float>> calcDis(std::vector<std::vector<float>> centroids)
	{
		std::vector<std::vector<float>> clalist;
		for (auto data : m_dataSet)
		{
			std::vector<std::vector<float>> diff(m_k);
			for (size_t i = 0; i < diff.size(); i++)
			{
				diff[i] = data;
			}
			for (size_t i = 0; i < diff.size(); i++)
			{
				for (size_t j = 0; j < diff[0].size(); j++)
				{
					diff[i][j] -= centroids[i][j];
					diff[i][j] = pow(diff[i][j], 2);
				}
			}
			std::vector<float> squaredDist(diff.size());
			for (size_t i = 0; i < diff.size(); i++)
			{
				for (size_t j = 0; j < diff[0].size(); j++)
				{
					squaredDist[i] += diff[i][j];
				}
				squaredDist[i] = sqrt(squaredDist[i]);
			}
			clalist.push_back(squaredDist);
		}

		//printMat(clalist);
		return clalist;
	}

	void classify(std::vector<std::vector<float>> centroids, std::vector<std::vector<float>>& newCentroids, std::vector<std::vector<float>>& changed)
	{
		std::vector<std::vector<float>> clalist = calcDis(centroids);
		std::vector<int> minDistIndices = getminDistIndices(clalist);

		newCentroids.resize(m_k, std::vector<float>(m_dataSet[0].size()));
		for (size_t i = 0; i < m_dataSet[0].size(); i++)
		{
			std::vector<float> sum(m_k);
			std::vector<int> num(m_k, 0);
			for (size_t j = 0; j < m_dataSet.size(); j++)
			{
				sum[minDistIndices[j]] += m_dataSet[j][i];
				++num[minDistIndices[j]];
			}

			for (size_t j = 0; j < m_k; j++)
			{
				//std::cout << sum[j] <<" "<<num[j] << std::endl;
				newCentroids[j][i] = sum[j] / num[j];
			}
		}

		//printMat(newCentroids);

		changed.resize(m_k, std::vector<float>(m_dataSet[0].size()));
		for (size_t i = 0; i < changed.size(); i++)
		{
			for (size_t j = 0; j < changed[0].size(); j++)
			{
				changed[i][j] = newCentroids[i][j] - centroids[i][j];
			}
		}
	}

	void predict(std::vector<std::vector<float>>& centroids, std::vector<std::vector<std::vector<float>>>& cluster)
	{
		srand((unsigned)time(NULL));
		std::vector<int> random_indices;
		while (random_indices.size() < m_k)
		{
			int random_index = rand() % m_dataSet.size();
			if(find(random_indices.begin(), random_indices.end(), random_index)== random_indices.end())
				random_indices.push_back(random_index);
		}

		centroids.resize(m_k, std::vector<float>(m_dataSet[0].size()));
		for (size_t i = 0; i < m_k; i++)
		{
			centroids[i] = m_dataSet[random_indices[i]];
		}

		std::vector<std::vector<float>> newCentroids;
		std::vector<std::vector<float>> changed;
		classify(centroids, newCentroids, changed);
		//printMat(centroids); printMat(newCentroids);

		while (!checkZeros(changed))
		{
			std::vector<std::vector<float>> copyCentroids = newCentroids;
			classify(copyCentroids, newCentroids, changed);
			//printMat(changed);
		}
		centroids = newCentroids;

		std::vector<std::vector<float>> clalist = calcDis(newCentroids);
		std::vector<int> minDistIndices = getminDistIndices(clalist);
		//for (auto i : minDistIndices)	std::cout << i << std::endl;

		cluster.resize(m_k);
		for (size_t i = 0; i < minDistIndices.size(); i++)
		{
			cluster[minDistIndices[i]].push_back(m_dataSet[i]);
		}
	}

private:
	std::vector<std::vector<float>> m_dataSet;
	int m_k;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> dataSet = { {1, 1},{1, 2},{2, 1},{6, 4},{6, 3},{5, 4} };
	//std::vector<std::vector<float>> centroids = { {1, 2},{6, 4} };
	int k = 2;
	KMeans kmeans = KMeans(dataSet, k);
	//kmeans.calcDis(centroids);
	//kmeans.classify(centroids);

	std::vector<std::vector<float>> centroids;
	std::vector<std::vector<std::vector<float>>> cluster;
	kmeans.predict(centroids, cluster);
	printMat(centroids);
	printMat(cluster[0]); printMat(cluster[1]);

	system("pause");
	return EXIT_SUCCESS;
}

