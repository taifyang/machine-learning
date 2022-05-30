#include <iostream>
#include <vector>
#include <time.h>

//感知机模型
class Pocket
{
public:
	Pocket(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		m_x = x;
		m_y = y;
		m_w.resize(m_x[0].size(), 0);
		m_best_w.resize(m_x[0].size(), 0);
		m_b = 0;
		m_best_b = 0;
	}

	float sign(std::vector<float> w, float b, std::vector<float> x)
	{
		float y = b;
		for (size_t i = 0; i < w.size(); i++)
		{
			y += w[i] * x[i];
		}
		return y;
	}

	std::vector<int> classify(std::vector<float> w, float b)
	{
		std::vector<int> mistakes;
		for (size_t i = 0; i < m_x.size(); i++)
		{
			float tmp_y = sign(w, b, m_x[i]);
			if (tmp_y*m_y[i] <= 0) //如果误分类
			{
				mistakes.push_back(i);
			}
		}
		return mistakes;
	}

	void update(float label_i, std::vector<float> data_i)
	{
		std::vector<float> tmp_w(m_w.size());
		for (size_t i = 0; i < m_w.size(); i++)
		{
			tmp_w[i] += label_i * data_i[i] + m_w[i];
		}
		float tmp_b = label_i + m_b;
		if (classify(m_best_w, m_best_b).size() >= classify(tmp_w, tmp_b).size())
		{
			for (size_t i = 0; i < m_w.size(); i++)
			{
				m_best_w[i] += label_i * data_i[i] + m_w[i];
			}
			m_best_b = label_i + m_b;
		}
		for (size_t i = 0; i < m_w.size(); i++)
		{
			m_w[i] = label_i * data_i[i] + m_w[i];
		}
		m_b = label_i + m_b;
	}

	void train(int max_iters)
	{
		int iters = 0;
		bool isFind = false;
		while (!isFind)
		{
			std::vector<int> mistakes = classify(m_w, m_b);
			if (mistakes.size() == 0)
			{
				std::cout << "最终训练得到的w为：";
				for (auto i : m_w)	std::cout << i << " ";
				std::cout << "\n最终训练得到的b为：";
				std::cout << m_b << "\n";
				break;
			}
			srand((int)time(0));
			int n = mistakes[rand() % (mistakes.size())];
			update(m_y[n], m_x[n]);
			++iters;
			if (iters == max_iters)
			{
				std::cout << "最终训练得到的w为：";
				for (auto i : m_w)	std::cout << i << " ";
				std::cout << "\n最终训练得到的b为：";
				std::cout << m_b << "\n";
				bool isFind = true;
			}
		}
	}

private:
	std::vector<std::vector<float>> m_x;
	std::vector<float> m_y;
	std::vector<float> m_w;
	std::vector<float> m_best_w;
	float m_b;
	float m_best_b;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 3, -3 },{ 4, -3 },{ 1, 1 },{ 1, 2 } };
	std::vector<float> y = { -1, -1, 1, 1 };

	Pocket mypocket = Pocket(x, y);
	mypocket.train(50);

	system("pause");
	return EXIT_SUCCESS;
}

