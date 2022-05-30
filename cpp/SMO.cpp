#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>

class SimpleSMO
{
public:
	SimpleSMO(std::vector<std::vector<float>> x, std::vector<float> y, float b, float c, float tolerance, int max_iter)
	{
		m_x = x;
		m_y = y;
		m_b = b;
		m_c = c;
		m_tolerance = tolerance;
		m_max_iter = max_iter;
		m_alpha.resize(m_x.size());
	}

	float g(std::vector<float> x_i)
	{
		std::vector<float> tmp_vec(m_x.size(), 0);
		for (size_t i = 0; i < m_x.size(); i++)
		{
			for (size_t j = 0; j < m_x[0].size(); j++)
			{
				tmp_vec[i] += m_x[i][j] * x_i[j];
			}
		}

		float tmp_val = 0;
		for (size_t i = 0; i < tmp_vec.size(); i++)
		{
			tmp_val += tmp_vec[i] * m_y[i];
		}

		float sum = 0;
		for (size_t i = 0; i < m_alpha.size(); i++)
		{
			sum += tmp_val*m_alpha[i];
		}

		return sum + m_b;
	}

	float Error(std::vector<float> x_i, float y_i)
	{
		return g(x_i) - y_i;
	}

	int SelectJ(int i)
	{
		srand((unsigned)time(NULL));
		int j = i;
		while (j == i)
		{
			j = rand() % m_x.size();
		}
		return j;
	}

	float Kernal(int m, int n)
	{
		float ret = 0;
		for (size_t i = 0; i < m_x[0].size(); i++)
		{
			ret += m_x[m][i] * m_x[n][i];
		}
		return ret;
	}

	void Optimization()
	{
		int iter = 0;
		while (iter < m_max_iter)
		{
			int alphaPairsChanged = 0;
			for (size_t i = 0; i < m_alpha.size(); i++)
			{
				float E_i = Error(m_x[i], m_y[i]);
				//std::cout << E_i << std::endl;
				if ((m_y[i] * E_i < -m_tolerance && m_alpha[i] < m_c) || (m_y[i] * E_i > m_tolerance && m_alpha[i] > 0))
				{
					int j = SelectJ(i);
					//std::cout << i << " " << j << std::endl;
					float E_j = Error(m_x[j], m_y[j]);
					float alpha_i_old = m_alpha[i];
					float alpha_j_old = m_alpha[j];

					float L, H;
					if (m_y[i] != m_y[j])
					{
						L = std::max(0.0f, alpha_j_old - alpha_i_old);
						H = std::min(m_c, m_c + alpha_j_old - alpha_i_old);
					}
					else {
						L = std::max(0.0f, alpha_j_old + alpha_i_old - m_c);
						H = std::min(m_c, alpha_j_old + alpha_i_old);
					}
					//std::cout << L << " " << H << std::endl;

					if (L == H)
					{
						std::cout << "L=H" << std::endl;
						continue;
					}

					float eta = 2 * Kernal(i, j) - Kernal(i, i) - Kernal(j, j);
					if (eta >= 0)
					{
						std::cout << "eta>=0" << std::endl;
						continue;
					}

					float alpha_j_new_unc = alpha_j_old - m_y[j] * (E_i - E_j) / eta;
					if (alpha_j_new_unc < L)
						m_alpha[j] = L;
					else if (alpha_j_new_unc > H)
						m_alpha[j] = H;
					else
						m_alpha[j] = alpha_j_new_unc;

					if (fabs(m_alpha[j] - alpha_j_old) < 0.00001)
					{
						std::cout << "j not moving enough" << std::endl;
						continue;
					}

					m_alpha[i] += m_y[i] * m_y[j] * (alpha_j_old - m_alpha[j]);

					float b_i_new = m_b - E_i - m_y[i] * Kernal(i, i)*(m_alpha[i] - alpha_i_old) - m_y[j] * Kernal(j, i)*(m_alpha[j] - alpha_j_old);
					float b_j_new = m_b - E_j - m_y[i] * Kernal(i, j)*(m_alpha[i] - alpha_i_old) - m_y[j] * Kernal(j, j)*(m_alpha[j] - alpha_j_old);

					if (m_alpha[i] > 0 && m_alpha[i] < m_c)
						m_b = b_i_new;
					else if (m_alpha[j] > 0 && m_alpha[j] < m_c)
						m_b = b_j_new;
					else
						m_b = (b_i_new + b_j_new) / 2.0;

					alphaPairsChanged += 1;
					std::cout << "External loop: " << iter << "; Internal loop i :" << i << "; alphaPairsChanged:" << alphaPairsChanged << std::endl;
				}
			}

			if (alphaPairsChanged == 0)
				iter += 1;
			else
				iter = 0;
			std::cout << "Iteration number: " << iter << std::endl;
		}
	}

private:
	std::vector<std::vector<float>> m_x;
	std::vector<float> m_y;
	float m_b;
	float m_c;
	float m_tolerance;
	int m_max_iter;
	std::vector<float> m_alpha;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 4,2 },{ 3,3 },{ 8,-2 },{ 2,-4 }, { 8,1 } };
	std::vector<float> y = { -1, -1, 1, -1, 1 };

	SimpleSMO smo = SimpleSMO(x, y, 0, 0.6, 0.001, 10);
	//std::cout << smo.Error(x[0], y[0]) << std::endl;
	smo.Optimization();

	system("pause");
	return EXIT_SUCCESS;
}