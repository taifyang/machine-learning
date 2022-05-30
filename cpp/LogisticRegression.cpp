#include <iostream>
#include <vector>
#include <Eigen/Dense>

//Logistic回归模型
class LogisticRegression
{
public:
	LogisticRegression(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		m_x.resize(x.size(), x[0].size());
		for (size_t i = 0; i < m_x.rows(); i++)
		{
			for (size_t j = 0; j < m_x.cols(); j++)
			{
				m_x(i, j) = x[i][j];
			}
		}

		m_y.resize(y.size(), 1);
		for (size_t i = 0; i <  m_y.rows(); i++)
		{
			m_y(i, 0) = y[i];
		}

		m_w = Eigen::MatrixXf::Zero(1, x[0].size());

		m_b = 0;
	}

	Eigen::VectorXf Logistic_sigmoid(Eigen::VectorXf y)
	{
		Eigen::VectorXf ret(y.size());
		for (size_t i = 0; i < y.size(); i++)
		{
			ret[i]= exp(y[i]) / (1 + exp(y[i]));
		}
		return ret;
	}

	float Logistic_cost(Eigen::VectorXf p, Eigen::VectorXf y)
	{
		float ret = 0.0;
		for (size_t i = 0; i < y.size(); i++)
		{
			ret += -y[i]*log(p[i]) - (1 - y[i])*log(1 - p[i]);
		}
		return ret;
	}

	float sum_vec(Eigen::VectorXf dz)
	{
		float sum = 0.0;
		for (size_t i = 0; i < dz.size(); i++)
		{
			sum += dz[i];
		}
		return sum;
	}

	void Logistic_BP(float learningrate, int iters)
	{
		for (size_t i = 0; i < iters; i++)
		{
			Eigen::VectorXf vec_tmp = m_x*m_w.transpose();
			Eigen::VectorXf vec_b(vec_tmp.size());
			for (size_t i = 0; i < vec_tmp.size(); i++)
			{
				vec_b[i] = m_b;
			}
			Eigen::VectorXf p = vec_tmp + vec_b;
			Eigen::VectorXf a = Logistic_sigmoid(p);

			std::cout << "iters: " << i << "  cost: " << Logistic_cost(a, m_y) << std::endl;

			Eigen::VectorXf dz = a - m_y;
			m_w -= learningrate*dz.transpose()*m_x;
			m_b -= learningrate*sum_vec(dz);
		}
		std::cout << "最终训练得到的w和b为：" << m_w << "  " << m_b << std::endl;
	}

	Eigen::VectorXf Logistic_predict(Eigen::MatrixXf x)
	{
		Eigen::VectorXf vec_tmp = x*m_w.transpose();
		Eigen::VectorXf vec_b(vec_tmp.size());
		for (size_t i = 0; i < vec_tmp.size(); i++)
		{
			vec_b[i] = m_b;
		}	
		
		Eigen::VectorXf pre = Logistic_sigmoid(vec_tmp + vec_b);
		return pre;
	}

private:
	Eigen::MatrixXf m_x;
	Eigen::MatrixXf m_y;
	Eigen::MatrixXf m_w;
	float m_b;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 0 },{ 1 },{ 2 },{ 3 } };
	std::vector<float> y = { 0, 0, 1, 1 };

	LogisticRegression logistic_regression = LogisticRegression(x, y); 
	logistic_regression.Logistic_BP(0.1, 100);
	Eigen::MatrixXf pre(1, 1);
	pre << 2.9;
	std::cout << "预测结果为：" << logistic_regression.Logistic_predict(pre) << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}

