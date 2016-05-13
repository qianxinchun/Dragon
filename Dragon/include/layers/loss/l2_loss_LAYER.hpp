#ifndef L2_LOSS_LAYER_HPP
#define L2_LOSS_LAYER_HPP

#include "loss_layer.hpp"

template <typename Dtype>
class L2LossLayer :public LossLayer < Dtype > {
public:
	L2LossLayer(const LayerParameter& param) :LossLayer<Dtype>(param) {}
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	Blob<Dtype> diff;

};

#endif