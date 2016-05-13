#ifndef DATA_TRANSFORMER_HPP
#define DATA_TRANSFORMER_HPP
#include <vector>

#include "protos/dragon.pb.h"
#include "blob.hpp"
#include "common.hpp"

#ifndef DISABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.hpp>
using namespace cv;
#endif

using namespace std;
template <typename Dtype>
class DataTransformer
{
public:
	DataTransformer(const TransformationParameter& param, Phase phase);
	vector<int> inferBlobShape(const Datum& datum);
	vector<int> inferBlobShape(const Mat& mat);
	void transform(const Datum& datum, Blob<Dtype>* shadow_blob);
	void transform(const Datum& datum, Dtype* shadow_data);
	void transform(const Mat& cv_img, Dtype* shadow_data);
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
