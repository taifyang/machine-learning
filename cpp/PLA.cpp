#include <iostream>
#include <vector>

/**
 * @description: 	感知机模型
 */
class Perceptron
{
public:
	/**
	 * @description: 	构造函数
	 * @param x			特征
	 * @param y			标签
	 */
	Perceptron(std::vector<std::vector<float>> x, std::vector<float> y)
	{
		m_x = x;
		m_y = y;
		m_w.resize(m_x[0].size(), 0);
		m_b = 0;
	}

	/**
	 * @description: 	计算y
	 * @param w			权重
	 * @param b			偏差
	 * @param x			x
	 * @return 			y
	 */
	float sign(std::vector<float> w, float b, std::vector<float> x)
	{
		float y = b;
		for (size_t i = 0; i < w.size(); i++)
		{
			y += w[i] * x[i];
		}
		return y;
	}

	/**
	 * @description: 	更新权重
	 * @param label_i	标签
	 * @param data_i	数据
	 */
	void update(float label_i, std::vector<float> data_i)
	{
		for (size_t i = 0; i < m_w.size(); i++)
		{
			m_w[i] += label_i * data_i[i];
		}
		m_b += label_i;
	}

	/**
	 * @description: 	训练
	 */
	void train()
	{
		bool isFind = false;
		while (!isFind)
		{
			float count = 0;
			for (size_t i = 0; i < m_x.size(); i++)
			{
				float tmp_y = sign(m_w, m_b, m_x[i]);
				if (tmp_y * m_y[i] <= 0)
				{
					++count;
					update(m_y[i], m_x[i]);
				}
			}
			if (count == 0)
			{
				std::cout << "最终训练得到的w为：";
				for (auto i : m_w)
					std::cout << i << " ";
				std::cout << "\n最终训练得到的b为：";
				std::cout << m_b << "\n";
				isFind = true;
			}
		}
	}

private:
	/**
	 * @description: 
	 */
	std::vector<std::vector<float>> m_x;

	/**
	 * @description: 
	 */
	std::vector<float> m_y;

	/**
	 * @description: 
	 */
	std::vector<float> m_w;

	/**
	 * @description: 
	 */
	float m_b;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{3, -3}, {4, -3}, {1, 1}, {1, 2}};
	std::vector<float> y = {-1, -1, 1, 1};
	Perceptron perceptron = Perceptron(x, y);
	perceptron.train();
	system("pause");
	return EXIT_SUCCESS;
}
