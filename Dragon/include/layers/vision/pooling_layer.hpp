# ifndef POOLING_LAYER_HPP
# define POOLING_LAYER_HPP

#include "../../layer.hpp"

template<typename Dtype>
class PoolingLayer :public Layer < Dtype > {
public:
	PoolingLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	int kernel_h, kernel_w;
	int stride_h, stride_w;
	int pad_h, pad_w;
	int channels, height, width;
	int pooling_height, pooling_width;
	bool global_pooling;
	Blob<Dtype> rand_idx;
	Blob<int> max_idx;
};


# endif