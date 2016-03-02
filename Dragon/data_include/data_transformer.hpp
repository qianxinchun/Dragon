#ifndef DATA_TRANSFORMER_HPP
#define DATA_TRANSFORMER_HPP
#include "include/dragon.pb.h"
#include "vector"
#include "include/blob.hpp"
#include "include/common.hpp"
using namespace std;
template <typename Dtype>
class DataTransformer
{
public:
	DataTransformer(const TransformationParameter& param, Phase phase);
	vector<int> inferBlobShape(const Datum& datum);
	void transform(const Datum& datum, Blob<Dtype>* shadow_blob);
	void transform(const Datum& datum, Dtype* shadow_data);
	void initRand();
	~DataTransformer() {}
	int rand(int n);
private:
	TransformationParameter param;
	Phase phase;
	Blob<Dtype> mean_blob;
	vector<Dtype> mean_vals;
	boost::shared_ptr<Dragon::RNG> ptr_rng;
};
#endif
