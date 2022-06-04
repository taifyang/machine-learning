#include <iostream>
#include <vector>
#include <Eigen/Dense>

/**
 * @description: 	逻辑斯蒂回归模型
 */
class LogisticRegression
{
public:
	/**
	 * @description: 	构造函数
	 * @param x			特征
	 * @param y			标签
	 */
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
		for (size_t i = 0; i < m_y.rows(); i++)
		{
			m_y(i, 0) = y[i];
		}

		m_w = Eigen::MatrixXf::Zero(1, x[0].size());

		m_b = 0;
	}

	/**
	 * @description: 	非线性层，将值域空间映射为(0, 1)
	 * @param y			标签
	 * @return 			映射
	 */
	Eigen::VectorXf Logistic_sigmoid(Eigen::VectorXf y)
	{
		Eigen::VectorXf ret(y.size());
		for (size_t i = 0; i < y.size(); i++)
		{
			ret[i] = exp(y[i]) / (1 + exp(y[i]));
		}
		return ret;
	}

	/**
	 * @description: 	损失函数
	 * @param p			概率
	 * @param y			标签
	 * @return 			损失
	 */
	float Logistic_cost(Eigen::VectorXf p, Eigen::VectorXf y)
	{
		float ret = 0.0;
		for (size_t i = 0; i < y.size(); i++)
		{
			ret += -y[i] * log(p[i]) - (1 - y[i]) * log(1 - p[i]);
		}
		return ret;
	}

	/**
	 * @description: 	向量求和
	 * @param dz		输入向量
	 * @return			向量元素之和
	 */
	float sum_vec(Eigen::VectorXf dz)
	{
		float sum = 0.0;
		for (size_t i = 0; i < dz.size(); i++)
		{
			sum += dz[i];
		}
		return sum;
	}

	/**
	 * @description: 		反向传播函数
	 * @param learningrate	学习率
	 * @param iters			迭代次数
	 */
	void Logistic_BP(float learningrate, int iters)
	{
		for (size_t i = 0; i < iters; i++)
		{
			Eigen::VectorXf vec_tmp = m_x * m_w.transpose();
			Eigen::VectorXf vec_b(vec_tmp.size());
			for (size_t i = 0; i < vec_tmp.size(); i++)
			{
				vec_b[i] = m_b;
			}
			Eigen::VectorXf p = vec_tmp + vec_b;
			Eigen::VectorXf a = Logistic_sigmoid(p);

			std::cout << "iters: " << i << "  cost: " << Logistic_cost(a, m_y) << std::endl;

			Eigen::VectorXf dz = a - m_y;
			m_w -= learningrate * dz.transpose() * m_x;
			m_b -= learningrate * sum_vec(dz);
		}
		std::cout << "最终训练得到的w和b为：" << m_w << "  " << m_b << std::endl;
	}

	/**
	 * @description: 	预测
	 * @param x			特征
	 * @return			预测值
	 */
	Eigen::VectorXf Logistic_predict(Eigen::MatrixXf x)
	{
		Eigen::VectorXf vec_tmp = x * m_w.transpose();
		Eigen::VectorXf vec_b(vec_tmp.size());
		for (size_t i = 0; i < vec_tmp.size(); i++)
		{
			vec_b[i] = m_b;
		}

		Eigen::VectorXf pre = Logistic_sigmoid(vec_tmp + vec_b);
		return pre;
	}

private:
	/**
	 * @description: 	特征
	 */
	Eigen::MatrixXf m_x;

	/**
	 * @description: 	标签
	 */
	Eigen::MatrixXf m_y;

	/**
	 * @description: 	权重
	 */
	Eigen::MatrixXf m_w;

	/**
	 * @description: 	偏差
	 */
	float m_b;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{0}, {1}, {2}, {3}};
	std::vector<float> y = {0, 0, 1, 1};
	LogisticRegression logistic_regression = LogisticRegression(x, y);
	logistic_regression.Logistic_BP(0.1, 100);
	Eigen::MatrixXf pre(1, 1);
	pre << 2.9;
	std::cout << "预测值为：" << logistic_regression.Logistic_predict(pre) << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}
