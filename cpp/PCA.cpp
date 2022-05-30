#include <iostream>
#include <vector>
#include <Eigen/Dense>

//计算协方差矩阵
Eigen::MatrixXf calc_cov(Eigen::MatrixXf mat)
{
	Eigen::MatrixXf meanVec = mat.colwise().mean();
	Eigen::RowVectorXf meanVecRow(Eigen::RowVectorXf::Map(meanVec.data(), mat.cols()));

	Eigen::MatrixXf zeroMeanMat = mat;
	zeroMeanMat.rowwise() -= meanVecRow;
	Eigen::MatrixXf covMat;
	if (mat.rows() == 1)
		covMat = (zeroMeanMat.adjoint()*zeroMeanMat) / float(mat.rows());
	else
		covMat = (zeroMeanMat.adjoint()*zeroMeanMat) / float(mat.rows() - 1);

	return covMat;
}

//返回从小到大排序的索引
std::vector<int> sort_index(std::vector<float> vec)
{
	std::vector<int>  vec_sorted(vec.size());
	for (size_t i = 0; i != vec_sorted.size(); ++i)
		vec_sorted[i] = i;
	std::sort(vec_sorted.begin(), vec_sorted.end(), [&vec](size_t i, size_t j) {return vec[i] >  vec[j]; });
	return vec_sorted;
}

class PCA
{
public:
	PCA(std::vector<std::vector<float>> x, int n_components = -1)
	{
		m_x.resize(x.size(), x[0].size());
		for (size_t i = 0; i < m_x.rows(); i++)
		{
			for (size_t j = 0; j < m_x.cols(); j++)
			{
				m_x(i, j) = x[i][j];
			}
		}

		m_dimension = x[0].size();

		if (n_components >= m_dimension)
			std::cout << "n_components error" << std::endl;

		m_components = n_components;
	}

	Eigen::MatrixXf get_feature()
	{
		Eigen::MatrixXf x_cov = calc_cov(m_x);
		Eigen::EigenSolver<Eigen::MatrixXf> eigen_solver(x_cov);
		Eigen::MatrixXcf eigen_value = eigen_solver.eigenvalues();
		Eigen::MatrixXcf eigen_vector = eigen_solver.eigenvectors();
		//std::cout << eigen_value << std::endl;
	    //std::cout << eigen_vector << std::endl;

		Eigen::MatrixXf c(eigen_value.rows(), eigen_vector.cols() + 1);
		std::vector<float> vec(eigen_value.rows());
		for (size_t i = 0; i < c.rows(); i++)
		{
			c(i, 0) = eigen_value(i, 0).real();
			vec[i] = c(i, 0);
			for (size_t j = 1; j < c.cols(); j++)
			{
				c(i, j) = eigen_vector(i, j - 1).real();
			}
		}
		//std::cout << c << std::endl;

		std::vector<int> index = sort_index(vec);
		Eigen::MatrixXf c_df_sort(c.rows(), c.cols());
		for (size_t i = 0; i < c.rows(); i++)
		{
			c_df_sort(i, 0) = c(index[i], 0);
			for (size_t j = 1; j < c.cols(); j++)
			{
				c_df_sort(i, j) = c(index[i], j);
			}
		}
		//std::cout << c_df_sort << std::endl;
		return c_df_sort;
	}

	Eigen::VectorXf explained_variance_()
	{
		Eigen::MatrixXf c_df_sort = get_feature();
		Eigen::VectorXf variance(c_df_sort.rows());
		for (size_t i = 0; i < variance.size(); i++)
		{
			variance[i] = c_df_sort(i, 0);
		}
		return variance;
	}

	Eigen::MatrixXf reduce_dimension()
	{
		Eigen::MatrixXf c_df_sort = get_feature();
		Eigen::VectorXf variance = explained_variance_();

		if (m_components != -1)
		{
			Eigen::MatrixXf p = c_df_sort.topRightCorner(m_components, c_df_sort.cols() - 1);
			Eigen::MatrixXf y = p*m_x.transpose();
			return y.transpose();		
		}

		float variance_sum = variance.sum();
		Eigen::VectorXf variance_radio = variance / variance_sum;
		float  variance_contribution = 0;

		int R = 0;
		for (; R < m_dimension; R++)
		{
			variance_contribution += variance_radio[R];
			if (variance_contribution >= 0.99)
				break;
		}

		Eigen::MatrixXf p = c_df_sort.topRightCorner(R + 1, c_df_sort.cols() - 1);
		Eigen::MatrixXf y = p*m_x.transpose();
		return y.transpose();
	}

private:
	Eigen::MatrixXf m_x;
	int m_dimension;
	int m_components;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { {-1, -1},{-2, -1},{-3, -2},{1, 1},{2, 1},{3, 2} };
	PCA pca = PCA(x);
	//pca.get_feature();
	std::cout << pca.reduce_dimension().transpose() << std::endl;
	std::cout << pca.explained_variance_().transpose() << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}

