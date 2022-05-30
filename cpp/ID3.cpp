#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

void printMat(std::vector<std::vector<float>> mat)
{
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (size_t j = 0; j < mat[0].size(); j++)
		{
			std::cout << mat[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

struct TreeNode
{
	float label;
	float value;	
	std::vector<TreeNode*> children;
	bool isleaf = false;
};

class ID3
{
public:
	ID3(std::vector<std::vector<float>> x, std::vector<float> y, std::vector<float> labels)
	{
		m_x = x;
		m_y = y;
		m_labels = labels;
		m_data = x;
		for (size_t i = 0; i < x.size(); i++)
		{
			m_data[i].push_back(y[i]);
		}
	}

	float calEntropy(std::vector<float> data)
	{
		float entropy = 0;
		std::map<float, int> c;
		for (auto i:data)	++c[i];
		int total = data.size();
		for (std::map<float, int>::iterator it = c.begin(); it != c.end(); it++)
		{
			entropy -= it->second / total*log(it->second / total);
		}
		return entropy;
	}
	
	std::vector<std::vector<float>> splitdata(std::vector<std::vector<float>> data, int col, float value)
	{
		std::vector<int> index;
		for (size_t i = 0; i < data.size(); i++)
		{
			if (data[i][col] == value)	index.push_back(i);
		}

		std::vector<std::vector<float>> data_r(index.size());
		for (size_t i = 0; i < index.size(); i++)
		{
			std::vector<float> vec;
			for (size_t j = 0; j < data[0].size(); j++)
			{
				if (j != col)	vec.push_back(data[index[i]][j]);
			}
			data_r[i] = vec;
		}
		return data_r;
	}

	int getBestFeature(std::vector<std::vector<float>> data)
	{
		std::vector<float> entropy_list;
		int numberAll = data.size();
		for (size_t col = 0; col < data[0].size()-1; col++)
		{
			float entropy_splited = 0;
			std::set<float> s;
			for (size_t i = 0; i < data.size(); i++)
			{
				s.insert(data[i][col]);
			}

			for (auto value : s)
			{
				//printMat(data);
				//std::cout << col << " " << value << std::endl;
				std::vector<std::vector<float>> splited = splitdata(data, col, value);
				//printMat(splited);
				std::vector<float> y_splited(splited.size());
				for (size_t i = 0; i < splited.size(); i++)
				{
					y_splited[i] = splited[i][splited[0].size() - 1];
				}			
				float entropy = calEntropy(y_splited);
				entropy_splited += y_splited.size() / numberAll*entropy;
			}
			entropy_list.push_back(entropy_splited);
		}

		int min_index = 0;
		float min_value = INT_MAX;
		for (size_t i = 0; i < entropy_list.size(); i++)
		{
			if (entropy_list[i] < min_value)
			{
				entropy_list[i] = min_value;
				min_index = i;
			}
		}
		return min_index;
	}

	TreeNode* CreateTree(std::vector<std::vector<float>> data, std::vector<float> label)
	{
		std::vector<float> feature_label = label;
		std::set<float> s1;
		for (size_t i = 0; i < data.size(); i++)
		{
			s1.insert(data[i][data[0].size() - 1]);
		}
		//for (auto i : s1)	std::cout << i << " "; std::cout << std::endl;
		if (s1.size() == 1)
		{
			TreeNode* node = new TreeNode;
			node->isleaf = true;
			node->label = data[0][data[0].size() - 1];
			//std::cout << node->label << std::endl;
			return node;
		}
			
		if (data[0].size() == 1)
		{
			std::map<float, int> c;
			for (auto i : data[data[0].size() - 1])	++c[i];

			std::vector<std::pair<float, int>>  c_vec;
			for (std::map<float, int>::iterator it = c.begin(); it != c.end(); it++)
			{
				c_vec.push_back(std::make_pair((*it).first, (*it).second));
			}
			std::sort(c_vec.begin(), c_vec.end(), [](std::pair<float, int> p1, std::pair<float, int> p2) {return p1.second < p2.second; });

			TreeNode* node = new TreeNode;
			node->isleaf = true;
			node->label = c_vec.rbegin()->first;
			return node;
		}

		int bestFeature = getBestFeature(data);
		float bestFeatureLabel = feature_label[bestFeature];
		//std::cout << bestFeature << " " << bestFeatureLabel << std::endl;

		TreeNode* node = new TreeNode;
		node->label = bestFeatureLabel;

		feature_label.erase(feature_label.begin() + bestFeature);

		std::set<float> s2;
		for (size_t i = 0; i < data.size(); i++)
		{
			s2.insert(data[i][bestFeatureLabel]);
		}
		//for (auto i : s2)	std::cout << i << " "; std::cout << std::endl;

		for (auto value : s2)
		{
			std::vector<float> sub_labels = feature_label;
			//for (auto i : sub_labels)	std::cout << i << " "; std::cout << std::endl;
			std::vector<std::vector<float>> splited_data = splitdata(data, bestFeature, value);
			//printMat(splited_data);

			TreeNode* childnode =  CreateTree(splited_data, sub_labels);
			childnode->value = value;
			node->children.push_back(childnode);
		}
		//std::cout << node->children.size() << std::endl;
		return node;
	}

	void fit()
	{
		m_tree = CreateTree(m_data, m_labels);
	}

	TreeNode* predict_vec(std::vector<float> vec, TreeNode* input_tree = nullptr)
	{
		if (input_tree == nullptr)
			input_tree = m_tree;

		int featureIndex;
		for (size_t i = 0; i < m_labels.size(); i++)
		{
			if (m_labels[i] == input_tree->label)
			{
				featureIndex = i;
				break;
			}
		}
		//std::cout << input_tree->label << " " << featureIndex << std::endl;
		
		std::vector<TreeNode*> secTree = (input_tree->children);
		//std::cout << input_tree->children.size() << std::endl;

		float vec_feature_val = vec[featureIndex];
		//std::cout << vec_feature_val << std::endl;

		bool isdict;
		size_t i = 0;
		for (; i < secTree.size(); i++)
		{
			if (secTree[i]->value == vec_feature_val)
			{
				isdict = !(secTree[i]->isleaf);
				break;
			}
		}

		if (!isdict)
			return secTree[i];
		else
			return predict_vec(vec, secTree[i]);
	}

	std::vector<float> predict(std::vector<std::vector<float>> x)
	{
		std::vector<float> out_put;
		for (auto i : x)
			out_put.push_back(predict_vec(i)->label);
		return out_put;
	}

private:
	std::vector<std::vector<float>> m_x;
	std::vector<float> m_y;
	std::vector<std::vector<float>> m_data;
	std::vector<float> m_labels;
public:
	TreeNode* m_tree;
};


int main(int argc, char* argv[])
{
	std::vector<std::vector<float>> x = { { 1, 1 },{ 1, 1 },{ 1, 0 },{ 0, 1 },{ 0, 1 } };
	std::vector<float> y = { { 1 },{ 1 },{ 0 },{ 0 },{ 0 } };
	std::vector<float> labels = { 0, 1 };

	ID3 id3 = ID3(x, y, labels);
	id3.fit();
	std::cout << "Ô¤²âÖµÎª£º" << id3.predict({ { 1, 0 } })[0] << std::endl;

	system("pause");
	return EXIT_SUCCESS;
}

