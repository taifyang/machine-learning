#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <Eigen/Dense>

/**
 * @description:	AdaBoost模型
 */
class AdaBoost
{
public:
	/**
	 * @description: 		单层决策树生成函数，通过阀值比较对数据进行分类，在阀值一边的数据分到类别-1，而在另一边的数据分到类别+1
	 * @param dataMatrix	数据集
	 * @param dimen			数据集列数
	 * @param threshVal		阈值
	 * @param threshIneq	比较方式
	 * @return 				分类结果
	 */
	Eigen::MatrixXf stumpClassify(Eigen::MatrixXf dataMatrix, int dimen, float threshVal, int threshIneq)
	{
		Eigen::MatrixXf retArray = Eigen::MatrixXf::Ones(dataMatrix.rows(), 1);
		if (threshIneq == 0)
		{
			std::vector<int> indices;
			for (size_t i = 0; i < dataMatrix.rows(); i++)
			{
				if (dataMatrix(i, dimen) <= threshVal)
					indices.push_back(i);
			}
			for (auto i : indices)
				retArray(i, 0) = -1.0;
		}
		else
		{
			std::vector<int> indices;
			for (size_t i = 0; i < dataMatrix.rows(); i++)
			{
				if (dataMatrix(i, dimen) > threshVal)
					indices.push_back(i);
			}
			for (auto i : indices)
				retArray(i, 0) = -1.0;
		}
		return retArray;
	}

	/**
	 * @description: 		遍历stumpClassify()函数所有的可能输入值，并找到数据集上的最佳的单层决策树
	 * @param dataArr		数据集
	 * @param classLabels	数据标签
	 * @param D				权重向量
	 */
	void buildStump(std::vector<std::vector<float>> dataArr, std::vector<float> classLabels, Eigen::MatrixXf D)
	{
		Eigen::MatrixXf dataMatrix(dataArr.size(), dataArr[0].size());
		for (size_t i = 0; i < dataMatrix.rows(); i++)
		{
			for (size_t j = 0; j < dataMatrix.cols(); j++)
			{
				dataMatrix(i, j) = dataArr[i][j];
			}
		}

		Eigen::MatrixXf labelMat(classLabels.size(), 1);
		for (size_t i = 0; i < labelMat.rows(); i++)
		{
			labelMat(i, 0) = classLabels[i];
		}

		int m = dataMatrix.rows(), n = dataMatrix.cols();
		float numSteps = 10.0;
		m_classEst = Eigen::MatrixXf::Zero(m, 1);
		m_minError = INT_MAX;

		for (size_t i = 0; i < n; i++)
		{
			std::vector<float> tmp(m);
			for (size_t j = 0; j < m; j++)
			{
				tmp[j] = dataMatrix(j, i);
			}
			float rangeMin = *std::min_element(tmp.begin(), tmp.end());
			float rangeMax = *std::max_element(tmp.begin(), tmp.end());
			float stepSize = (rangeMax - rangeMin) / numSteps;

			for (int j = -1; j < int(numSteps) + 1; j++)
			{
				for (auto inequal : {0, 1})
				{
					float threshVal = rangeMin + j * stepSize;
					Eigen::MatrixXf predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal);
					Eigen::MatrixXf errArr = Eigen::MatrixXf::Ones(m, 1);
					for (size_t k = 0; k < m; k++)
					{
						if (predictedVals(k, 0) == labelMat(k, 0))
							errArr(k, 0) = 0;
					}
					float weightedError = (D.transpose() * errArr)(0, 0);

					if (weightedError < m_minError)
					{
						m_minError = weightedError;
						m_classEst = predictedVals;
						m_bestStump["dim"] = i;
						m_bestStump["thresh"] = threshVal;
						m_bestStump["ineq"] = inequal;
					}
				}
			}
		}
	}

	/**
	 * @description:	完整AdaBoost算法实现
	 * @param x     	特征
	 * @param y     	类别标签
	 * @param iters 	迭代次数
	 */
	void fit(std::vector<std::vector<float>> x, std::vector<float> y, int iters = 10)
	{
		int m = x.size();
		Eigen::MatrixXf D = Eigen::MatrixXf::Ones(m, 1) / m;
		Eigen::MatrixXf aggClassEst = Eigen::MatrixXf::Zero(m, 1);

		for (size_t i = 0; i < iters; i++)
		{
			buildStump(x, y, D);
			float alpha = 0.5 * log((1.0 - m_minError) / std::max(m_minError, 1e-16f));
			m_bestStump["alpha"] = alpha;
			m_weakClassArr.push_back(m_bestStump);

			Eigen::MatrixXf labelMat(y.size(), 1);
			for (size_t i = 0; i < labelMat.rows(); i++)
			{
				labelMat(i, 0) = y[i];
			}

			Eigen::MatrixXf expon = (-alpha * labelMat).cwiseProduct(m_classEst);
			for (size_t i = 0; i < expon.rows(); i++)
			{
				expon(i, 0) = exp(expon(i, 0));
			}
			D = D.cwiseProduct(expon);
			D = D / D.sum();

			aggClassEst += alpha * m_classEst;

			Eigen::MatrixXf mul1(aggClassEst.rows(), 1), mul2 = Eigen::MatrixXf::Ones(m, 1);
			for (size_t i = 0; i < aggClassEst.rows(); i++)
			{
				if (aggClassEst(i, 0) * labelMat(i, 0) < 0)
					mul1(i, 0) = 1;
				else
					mul1(i, 0) = 0;
			}

			Eigen::MatrixXf aggErrors = mul1.cwiseProduct(mul2);
			float errorRate = aggErrors.sum() / m;

			if (errorRate == 0.0)
				break;
		}
	}

	/**
	 * @description:    预测
	 * @param x         待分类样本
	 * @return          分类结果
	 */
	std::vector<float> predict(std::vector<std::vector<float>> x)
	{
		Eigen::MatrixXf dataMatrix(x.size(), x[0].size());
		for (size_t i = 0; i < dataMatrix.rows(); i++)
		{
			for (size_t j = 0; j < dataMatrix.cols(); j++)
			{
				dataMatrix(i, j) = x[i][j];
			}
		}
		int m = dataMatrix.rows();
		Eigen::MatrixXf aggClassEst = Eigen::MatrixXf::Zero(m, 1);

		for (size_t i = 0; i < m_weakClassArr.size(); i++)
		{
			Eigen::MatrixXf classEst = stumpClassify(dataMatrix, m_weakClassArr[i]["dim"], m_weakClassArr[i]["thresh"], m_weakClassArr[i]["ineq"]);
			aggClassEst += m_weakClassArr[i]["alpha"] * classEst;
		}

		std::vector<float> ret(aggClassEst.rows());
		for (size_t i = 0; i < ret.size(); i++)
		{
			if (aggClassEst(i, 0) > 0)
				ret[i] = 1;
			else
				ret[i] = -1;
		}
		return ret;
	}

private:
	/**
	 * @description:	存储给定权重向量D时所得到的最佳决策树的相关信息
	 */
	std::map<std::string, float> m_bestStump;

	/**
	 * @description:	类别估计值
	 */
	Eigen::MatrixXf m_classEst;

	/**
	 * @description:	最小错误率
	 */
	float m_minError;

	/**
	 * @description:	 存储单层决策树的信息
	 */
	std::vector<std::map<std::string, float>> m_weakClassArr;
};

int main(int argc, char *argv[])
{
	std::vector<std::vector<float>> x = {{1.0f, 2.1f}, {2.0f, 1.1f}, {1.3f, 1.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}};
	std::vector<float> y = {1, 1, -1, -1, 1};
	AdaBoost ada = AdaBoost();
	Eigen::MatrixXf D = Eigen::MatrixXf::Ones(5, 1) / 5.0;
	ada.buildStump(x, y, D);
	ada.fit(x, y, 9);
	std::cout << "预测值为：" << ada.predict({{0, 0}})[0] << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}
