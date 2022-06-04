#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <iterator>

#define PI 3.1415926

/**
 * @description: 	高斯朴素贝叶斯模型
 */
class GaussionNB
{
public:
	/**
	 * @description: 	构造函数
	 * @param fit_prior	是否学习类的先验几率，False则使用统一的先验
	 */
	GaussionNB(bool fit_prior = true) : m_fit_prior(fit_prior) {}

	/**
	 * @description: 	建立模型
	 * @param x			特征
	 * @param y			标签
	 */
	void fit(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		for (auto i : y)
			m_classes.insert(i);
		if (m_class_prior.size() == 0)
		{
			int class_num = m_classes.size();
			if (!m_fit_prior)
			{
				for (auto d : m_classes)
					m_class_prior[d] = 1.0 / class_num;
			}
			else
			{
				for (auto d : m_classes)
				{
					float c_num = 0;
					for (auto i : y)
						c_num += (d == i ? 1 : 0);
					m_class_prior[d] = c_num / y.size();
				}
			}
		}

		for (auto pa : m_class_prior)
		{
			std::vector<int> y_index;
			for (size_t i = 0; i < y.size(); i++)
			{
				if (y[i] == pa.first)
					y_index.push_back(i);
			}

			for (size_t i = 0; i < x.size(); i++)
			{
				std::vector<float> x_class;

				for (auto j : y_index)
				{
					x_class.push_back(x[i][j]);

					std::string pkey = std::to_string(i) + "|" + std::to_string(pa.first);

					float mean_val = 0;
					for (size_t k = 0; k < x_class.size(); k++)
					{
						mean_val += x_class[k];
					}
					mean_val /= x_class.size();
					m_mean[pkey] = mean_val;

					float var_val = 0;
					for (size_t k = 0; k < x_class.size(); k++)
					{
						var_val += pow(x_class[k] - mean_val, 2);
					}
					var_val /= x_class.size();
					m_var[pkey] = var_val;
				}
			}
		}
	}

	/**
	 * @description: 	计算高斯概率
	 * @param mu		均值
	 * @param sigma		方差
	 * @param x			特征
	 * @return			高斯概率
	 */
	float _calculat_prob_gaussion(float mu, float sigma, float x)
	{
		return 1.0 / (sigma * sqrt(2 * PI)) * exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)));
	}

	/**
	 * @description: 	预测
	 * @param x			特征
	 * @return			预测值
	 */
	std::vector<float> predict(std::vector<std::vector<float>> x)
	{
		std::vector<float> labels;
		for (size_t i = 0; i < x.size(); i++)
		{
			m_predict_prob.clear();
			for (auto pa : m_class_prior)
			{
				m_predict_prob[pa.first] = m_class_prior[pa.first];
				for (size_t j = 0; j < x[i].size(); ++j)
				{
					std::string tkey = std::to_string(j) + "|" + std::to_string(pa.first);

					float mu = m_mean[tkey];
					float sigma = m_var[tkey];
					m_predict_prob[pa.first] *= _calculat_prob_gaussion(mu, sigma, x[i][j]);
				}
			}

			std::vector<std::pair<float, float>> m_predict_prob_vec;
			for (std::map<float, float>::iterator it = m_predict_prob.begin(); it != m_predict_prob.end(); it++)
			{
				m_predict_prob_vec.push_back(std::make_pair((*it).first, (*it).second));
			}
			std::sort(m_predict_prob_vec.begin(), m_predict_prob_vec.end(), [](std::pair<float, float> p1, std::pair<float, float> p2) { return p1.second < p2.second; });
			float label = m_predict_prob_vec.rbegin()->first;
			labels.push_back(label);
		}

		return labels;
	}

private:
	/**
	 * @description: 	是否学习类的先验几率，False则使用统一的先验
	 */
	bool m_fit_prior;

	/**
	 * @description: 	类的先验几率，若指定则先验不能根据数据调整  
	 */
	std::map<float, float> m_class_prior;

	/**
	 * @description: 	类别
	 */
	std::set<float> m_classes;

	/**
	 * @description: 	均值
	 */
	std::map<std::string, float> m_mean;

	/**	
	 * @description: 	方差
	 */
	std::map<std::string, float> m_var;

	/**
	 * @description: 	预测概率
	 */
	std::map<float, float> m_predict_prob;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3}, {1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 3}};
	std::vector<float> y = {-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1};
	GaussionNB gnb = GaussionNB();
	gnb.fit(x, y);
	std::cout << "预测值为：" << gnb.predict({{2, 1}})[0] << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}
