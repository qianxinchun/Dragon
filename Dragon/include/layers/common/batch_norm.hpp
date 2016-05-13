# ifndef BATCH_NORM_LAYER_HPP
# define BATCH_NORM_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class BatchNormLayer :public Layer < Dtype > {
public:
	BatchNormLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	Blob<Dtype> mean, var, temp, x_norm, expand_var;
	bool use_global_stats;
	Dtype decay, eps;
	int channels;
	Blob<Dtype> batch_sum_multiplier, spatial_sum_multiplier;
	Blob<Dtype> num_by_channels;
};

# endif