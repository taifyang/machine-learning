#include <iostream>
#include <vector>

//感知机模型
class Perceptron
{
public:
	Perceptron(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		m_x = x;
		m_y = y;
		m_w.resize(m_x[0].size(), 0);
		m_b = 0;
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

	void update(float label_i, std::vector<float> data_i)
	{
		for (size_t i = 0; i < m_w.size(); i++)
		{
			m_w[i] += label_i * data_i[i];
		}
		m_b += label_i;
	}

	void train()
	{
		bool isFind = false;
		while (!isFind)
		{
			float count = 0;
			for (size_t i = 0; i < m_x.size(); i++)
			{
				float tmp_y = sign(m_w, m_b, m_x[i]);
				if (tmp_y*m_y[i] <= 0) //如果误分类
				{
					++count;
					update(m_y[i], m_x[i]);
				}
			}
			if (count == 0)
			{
				std::cout << "最终训练得到的w为：";
				for (auto i : m_w)	std::cout << i << " ";
				std::cout << "\n最终训练得到的b为：";
				std::cout << m_b << "\n";
				isFind = true;
			}
		}
	}

private:
	std::vector<std::vector<float>> m_x;
	std::vector<float> m_y;
	std::vector<float> m_w;
	float m_b;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 3, -3 },{ 4, -3 },{ 1, 1 },{ 1, 2 } };
	std::vector<float> y = { -1, -1, 1, 1 };

	Perceptron myperceptron = Perceptron(x, y);
	myperceptron.train();

	system("pause");
	return EXIT_SUCCESS;
}
