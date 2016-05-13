#ifndef SOFTMAXLOSS_LAYER_HPP
#define SOFTMAXLOSS_LAYER_HPP

#include "loss_layer.hpp"

template <typename Dtype>
class SoftmaxWithLossLayer :public LossLayer < Dtype > {
public:
	SoftmaxWithLossLayer(const LayerParameter& param) :LossLayer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	boost::shared_ptr<Layer<Dtype>> softmax_layer;
	vector<Blob<Dtype>*> softmax_bottom, softmax_top;
	Blob<Dtype> prob;
	bool need_norm;
	int axis, outer_num, inner_num, ignore_label;
	bool has_ignore_label, has_normalize;
};

# endif