#include <iostream>
#include <vector>
#include <random>
#include <ctime>

class EM
{
public:
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

	void e_step()
	{
		for (size_t i = 0; i < m_n; i++)
		{
			float denom = 0;
			for (size_t j = 0; j < m_k; j++)
			{
				denom += exp((-1 / (2 * pow(m_sigma, 2)))*pow(m_x[0][i] - m_mu[j],2));
			}
			for (size_t j = 0; j < m_k; j++)
			{
				float numer = exp((-1 / (2 * pow(m_sigma, 2)))*pow(m_x[0][i] - m_mu[j], 2));
				m_expectations[i][j] = numer / denom;
			}
		}
	}

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

	void predict(int iter_num, float epsilon)
	{
		for (size_t i = 0; i < iter_num; i++)
		{
			std::vector<float> old_mu = m_mu;
			e_step();
			m_step();
			for (auto j : m_mu) std::cout << j << " "; std::cout << std::endl;

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
	std::vector<std::vector<float>> m_x;
	float m_sigma;
	int m_k;
	int m_n;
	std::vector<float> m_mu;
	std::vector<std::vector<float>> m_expectations;
};

int main(int argc, char* argv[])
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
		else {
			std::normal_distribution<double> dist(mu2, sigma);
			x[0][i] = dist(generator);
		}
		//std::cout << x[0][i] << std::endl;
	}
	
	EM em = EM(x, sigma, k, n);
	em.predict(100, 0.001);

	system("pause");
	return EXIT_SUCCESS;
}

