#ifndef RELU_LAYER_HPP
#define RELU_LAYER_HPP

#include "neuron_layer.hpp"

template <typename Dtype>
class ReLULayer :public NeuronLayer < Dtype > {
public:
	ReLULayer(const LayerParameter& param) :NeuronLayer<Dtype>(param) {}
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
};

# endif