#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <iterator>

class BernoulliNB
{
public:
	BernoulliNB(float alpha, bool fit_prior = true) :m_alpha(alpha), m_fit_prior(fit_prior) {}

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
			else {
				for (auto d : m_classes)
				{
					int c_num = 0;
					for (auto i : y)	c_num += (d == i ? 1 : 0);
					m_class_prior[d] = (c_num + m_alpha) / (y.size() + class_num*m_alpha);
				}
			}
		}

		std::vector<float> x_class = { 0, 1 };
		for (auto pa : m_class_prior)
		{
			//	std::cout << pa.first << " " << pa.second << std::endl;
			std::vector<int> y_index;
			for (size_t i = 0; i < y.size(); i++)
			{
				if (y[i] == pa.first)	y_index.push_back(i);
			}

			for (size_t i = 0; i < x.size(); i++)
			{
				for (auto c : x_class)
				{
					std::vector<int> x_index;
					for (size_t j = 0; j < x[i].size(); j++)
					{
						if (x[i][j] == c)	x_index.push_back(j);
					}

					std::sort(x_index.begin(), x_index.end());
					std::sort(y_index.begin(), y_index.end());
					std::vector<int> xy_index;
					std::set_intersection(x_index.begin(), x_index.end(), y_index.begin(), y_index.end(), std::back_inserter(xy_index));//求交集 

					std::string pkey = std::to_string(c) + "|" + std::to_string(pa.first);
					//std::cout << pkey << std::endl;

					m_conditional_prob[pkey] = (xy_index.size() + m_alpha) / (y_index.size() + 2);
				}
			}
		}

		for (auto pa : m_conditional_prob)
			std::cout << pa.first << " " << pa.second << std::endl;
	}

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

			for (auto pa : m_predict_prob)
				std::cout << pa.first << " " << pa.second << std::endl;

			std::vector<std::pair<float, float>>  m_predict_prob_vec;
			for (std::map<float, float>::iterator it = m_predict_prob.begin(); it != m_predict_prob.end(); it++)
			{
				m_predict_prob_vec.push_back(std::make_pair((*it).first, (*it).second));
			}
			std::sort(m_predict_prob_vec.begin(), m_predict_prob_vec.end(), [](std::pair<float, float> p1, std::pair<float, float> p2) {return p1.second < p2.second; });
			float label = m_predict_prob_vec.rbegin()->first;
			labels.push_back(label);
		}

		return labels;
	}

private:
	float m_alpha;
	bool m_fit_prior;
	std::map<float, float> m_class_prior;
	std::set<float> m_classes;
	std::map<std::string, float> m_conditional_prob;
	std::map<float, float> m_predict_prob;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 1,1,1,1,1,0,0,0,0,0,1,1,1,1,1 },{ 0,1,1,0,0,0,1,1,1,1,1,1,1,1,1 } };
	std::vector<float> y = { -1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1 };

	BernoulliNB bnb = BernoulliNB(1.0, true);
	bnb.fit(x, y);
	std::cout << "预测值为：" << bnb.predict({ { 0,0 } })[0] << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}