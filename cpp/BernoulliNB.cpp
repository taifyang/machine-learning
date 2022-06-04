#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <iterator>

/**
 * @description: 	伯努利朴素贝叶斯模型
 */
class BernoulliNB
{
public:
	/**
	 * @description: 	构造函数
	 * @param alpha		平滑系数
	 * @param fit_prior	是否学习类的先验几率，False则使用统一的先验
	 */
	BernoulliNB(float alpha, bool fit_prior = true) : m_alpha(alpha), m_fit_prior(fit_prior) {}

	/**
	 * @description:	计算类别y的先验几率
	 * @param x			特征
	 * @param y			类别
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
					int c_num = 0;
					for (auto i : y)
						c_num += (d == i ? 1 : 0);
					m_class_prior[d] = (c_num + m_alpha) / (y.size() + class_num * m_alpha);
				}
			}
		}

		std::vector<float> x_class = {0, 1};
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
				for (auto c : x_class)
				{
					std::vector<int> x_index;
					for (size_t j = 0; j < x[i].size(); j++)
					{
						if (x[i][j] == c)
							x_index.push_back(j);
					}

					std::sort(x_index.begin(), x_index.end());
					std::sort(y_index.begin(), y_index.end());
					std::vector<int> xy_index;
					std::set_intersection(x_index.begin(), x_index.end(), y_index.begin(), y_index.end(), std::back_inserter(xy_index)); //�󽻼�

					std::string pkey = std::to_string(c) + "|" + std::to_string(pa.first);
					m_conditional_prob[pkey] = (xy_index.size() + m_alpha) / (y_index.size() + 2);
				}
			}
		}
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
			for (auto j : m_classes)
			{
				m_predict_prob[j] = m_class_prior[j];
				for (auto d : x[i])
				{
					m_predict_prob[j] *= m_conditional_prob[std::to_string(d) + "|" + std::to_string(j)];
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
	 * @description: 	平滑系数	
	 */
	float m_alpha;

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
	 * @description: 	条件概率
	 */
	std::map<std::string, float> m_conditional_prob;

	/**
	 * @description: 	预测概率
	 */
	std::map<float, float> m_predict_prob;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}, {0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
	std::vector<float> y = {-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1};
	BernoulliNB bnb = BernoulliNB(1.0, true);
	bnb.fit(x, y);
	std::cout << "预测值为：" << bnb.predict({{0, 0}})[0] << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}
