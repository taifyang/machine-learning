#include <iostream>
#include <vector>
#include <Eigen/Dense>

//线性回归模型
void LinearRegression(std::vector<std::vector<float>> x, std::vector<float> y, float alpha, int iters)
{
	for (size_t i = 0; i < x.size(); i++)
	{
		x[i].insert(x[i].begin(), 1);
	}
	Eigen::MatrixXf mat_x(x.size(), x[0].size());
	for (size_t i = 0; i < mat_x.rows(); i++)
	{
		for (size_t j = 0; j < mat_x.cols(); j++)
		{
			mat_x(i, j) = x[i][j];
		}
	}

	Eigen::MatrixXf mat_y(y.size(), 1);
	for (size_t i = 0; i < mat_y.rows(); i++)
	{
		mat_y(i, 0) = y[i];
	}

	Eigen::MatrixXf mat_w = Eigen::MatrixXf::Zero(1, x[0].size());

	for (size_t i = 0; i < iters; i++)
	{
		mat_w -= (alpha / y.size())*(mat_x*mat_w.transpose() - mat_y).transpose()*mat_x;
		std::cout <<"iters: " << i << " cost: " << 1.0 / (2 * y.size())*(mat_x*mat_w.transpose() - mat_y).squaredNorm() << std::endl;
	}
	std::cout << "最终训练得到的w为：" << mat_w << std::endl;
}


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 1 },{ 2 },{ 3 },{ 4 } };
	std::vector<float> y = { 1, 2, 2.9f, 4.1f };

	LinearRegression(x, y, 0.1, 100);

	system("pause");
	return EXIT_SUCCESS;
}
