# ifndef SOFTMAX_LAYER_HPP
# define SOFTMAX_LAYER_HPP

#include "../../layer.hpp"

template<typename Dtype>
class SoftmaxLayer :public Layer < Dtype > {
public:
	SoftmaxLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	int outer_num, inner_num, axis;
	Blob<Dtype> sum_multiplier;
	Blob<Dtype> scale;
};

# endif