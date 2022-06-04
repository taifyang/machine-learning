#include <iostream>
#include <vector>
#include <random>
#include <ctime>

/**
 * @description: 	EM模型
 */
class EM
{
public:
	/**
	 * @description: 	构造函数
	 * @param x			特征
	 * @param sigma		高斯分布均方差
	 * @param k			高斯混合模型数
	 * @param n			数据个数
	 */
	EM(std::vector<std::vector<float>> x, float sigma, int k, int n)
	{
		m_x = x;
		m_sigma = sigma;
		m_k = k;
		m_n = n;

		m_mu.resize(2);
		for (size_t i = 0; i < m_mu.size(); i++)
		{
			m_mu[i] = 2 * rand() / double(RAND_MAX);
		}

		m_expectations.resize(n, std::vector<float>(k));
		for (size_t i = 0; i < n; i++)
		{
			for (size_t j = 0; j < k; j++)
			{
				m_expectations[i][j] = 0;
			}
		}
	}

	/**
	 * @description: 	EM算法步骤1，计算E[zij]
	 */
	void e_step()
	{
		for (size_t i = 0; i < m_n; i++)
		{
			float denom = 0;
			for (size_t j = 0; j < m_k; j++)
			{
				denom += exp((-1 / (2 * pow(m_sigma, 2))) * pow(m_x[0][i] - m_mu[j], 2));
			}
			for (size_t j = 0; j < m_k; j++)
			{
				float numer = exp((-1 / (2 * pow(m_sigma, 2))) * pow(m_x[0][i] - m_mu[j], 2));
				m_expectations[i][j] = numer / denom;
			}
		}
	}

	/**
	 * @description: 	EM算法步骤2，求最大化E[zij]的参数mu
	 */
	void m_step()
	{
		for (size_t j = 0; j < m_k; j++)
		{
			float numer = 0;
			float denom = 0;
			for (size_t i = 0; i < m_n; i++)
			{
				numer += m_expectations[i][j] * m_x[0][i];
				denom += m_expectations[i][j];
				m_mu[j] = numer / denom;
			}
		}
	}

	/**
	 * @description: 	预测
	 * @param iter_num	迭代次数
	 * @param epsilon	精度
	 */
	void predict(int iter_num, float epsilon)
	{
		for (size_t i = 0; i < iter_num; i++)
		{
			std::vector<float> old_mu = m_mu;
			e_step();
			m_step();
			for (auto j : m_mu)
				std::cout << j << " ";
			std::cout << std::endl;

			float error = 0;
			for (size_t j = 0; j < m_mu.size(); j++)
			{
				error += fabs(m_mu[j] - old_mu[j]);
			}
			if (error < epsilon)
				break;
		}
	}

private:
	/**
	 * @description: 	特征
	 */
	std::vector<std::vector<float>> m_x;

	/**
	 * @description: 	高斯分布均方差
	 */
	float m_sigma;

	/**
	 * @description: 	高斯混合模型数
	 */
	int m_k;

	/**
	 * @description: 	数据个数
	 */
	int m_n;

	/**
	 * @description: 	高斯分布均值
	 */
	std::vector<float> m_mu;

	/**
	 * @description: 	期望
	 */
	std::vector<std::vector<float>> m_expectations;
};

int main(int argc, char *argv[])
{
	float sigma = 6;
	float mu1 = 40;
	float mu2 = 20;
	int n = 100;
	int k = 2;

	std::vector<std::vector<float>> x(1, std::vector<float>(n));
	srand((unsigned)time(NULL));
	std::default_random_engine generator;
	for (size_t i = 0; i < n; i++)
	{
		if (rand() / double(RAND_MAX) > 0.5)
		{
			std::normal_distribution<double> dist(mu1, sigma);
			x[0][i] = dist(generator);
		}
		else
		{
			std::normal_distribution<double> dist(mu2, sigma);
			x[0][i] = dist(generator);
		}
	}

	EM em = EM(x, sigma, k, n);
	em.predict(100, 0.001);

	system("pause");
	return EXIT_SUCCESS;
}
