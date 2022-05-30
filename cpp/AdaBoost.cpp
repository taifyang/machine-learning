#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <Eigen/Dense>

class AdaBoost
{
public:
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
			for (auto i : indices)	retArray(i, 0) = -1.0;
		}
		else {
			std::vector<int> indices;
			for (size_t i = 0; i < dataMatrix.rows(); i++)
			{
				if (dataMatrix(i, dimen) > threshVal)
					indices.push_back(i);
			}
			for (auto i : indices)	retArray(i, 0) = -1.0;
		}
		return retArray;
	}

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
		m_error = INT_MAX;

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
				for (auto inequal : { 0, 1 })
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
					std::cout << "i:" << i << " threshVal:" << threshVal << " inequal:" << inequal << " weightedError:" << weightedError << std::endl;
					if (weightedError < m_error)
					{
						m_error = weightedError;
						m_classEst = predictedVals;
						m_bestStump["dim"] = i;
						m_bestStump["thresh"] = threshVal;
						m_bestStump["ineq"] = inequal;
					}
				}
			}
		}
	}

	void fit(std::vector<std::vector<float>> x, std::vector<float> y, int iters = 10)
	{
		int m = x.size();
		Eigen::MatrixXf D = Eigen::MatrixXf::Ones(m, 1) / m;
		Eigen::MatrixXf aggClassEst = Eigen::MatrixXf::Zero(m, 1);

		for (size_t i = 0; i < iters; i++)
		{
			buildStump(x, y, D);
			//std::cout << "D: " << D.transpose() << std::endl;

			float alpha = 0.5 * log((1.0 - m_error) / std::max(m_error, 1e-16f));
			//std::cout << "alpha: " << alpha << std::endl;
			m_bestStump["alpha"] = alpha;
			m_classifierArr.push_back(m_bestStump);
			//std::cout << "classEst: " << classEst<< std::endl;

			Eigen::MatrixXf labelMat(y.size(), 1);
			for (size_t i = 0; i < labelMat.rows(); i++)
			{
				labelMat(i, 0) = y[i];
			}
			//std::cout << "labelMat: " << labelMat << std::endl;
			Eigen::MatrixXf expon = (-alpha *labelMat).cwiseProduct(m_classEst);
			//std::cout << "expon: " << expon << std::endl;

			for (size_t i = 0; i < expon.rows(); i++)
			{
				expon(i, 0) = exp(expon(i, 0));
			}
			D = D.cwiseProduct(expon);
			D = D / D.sum();
			//std::cout << "D: " << D << std::endl;

			aggClassEst += alpha * m_classEst;
			//std::cout << "aggClassEst: " << aggClassEst << std::endl;

			Eigen::MatrixXf mul1(aggClassEst.rows(), 1), mul2 = Eigen::MatrixXf::Ones(m, 1);
			for (size_t i = 0; i < aggClassEst.rows(); i++)
			{
				if (aggClassEst(i, 0)*labelMat(i, 0) < 0)
					mul1(i, 0) = 1;
				else
					mul1(i, 0) = 0;
			}
			//std::cout << "mul1: " << mul1 << std::endl;

			Eigen::MatrixXf aggErrors = mul1.cwiseProduct(mul2);
			float errorRate = aggErrors.sum() / m;
			std::cout << "errorRate: " << errorRate << std::endl;

			if (errorRate == 0.0)
				break;
		}
	}

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

		for (size_t i = 0; i < m_classifierArr.size(); i++)
		{
			Eigen::MatrixXf classEst = stumpClassify(dataMatrix, m_classifierArr[i]["dim"], m_classifierArr[i]["thresh"], m_classifierArr[i]["ineq"]);
			aggClassEst += m_classifierArr[i]["alpha"] * classEst;
			std::cout << "aggClassEst: " << aggClassEst << std::endl;
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
	std::map<std::string, float> m_bestStump;
	Eigen::MatrixXf m_classEst;
	float m_error;
	std::vector<std::map<std::string, float>> m_classifierArr;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 1.0f, 2.1f},{ 2.0f, 1.1f },{ 1.3f, 1.0f },{ 1.0f, 1.0f } ,{ 2.0f, 1.0f} };
	std::vector<float> y = { 1, 1, -1, -1, 1 };
	AdaBoost ada = AdaBoost();

	Eigen::MatrixXf D = Eigen::MatrixXf::Ones(5, 1) / 5.0;
	
	std::cout << "最佳单层决策树相关信息：";
	ada.buildStump(x, y, D);

	ada.fit(x, y, 9);
	std::cout << "预测值为：" << ada.predict({ {0, 0} })[0] << std::endl;
	system("pause");
	return EXIT_SUCCESS;
}

