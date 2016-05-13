#ifndef POWER_LAYER_HPP
#define POWER_LAYER_HPP

#include "neuron_layer.hpp"

template <typename Dtype>
class PowerLayer : public NeuronLayer<Dtype> {
public:
	PowerLayer(const LayerParameter& param) : NeuronLayer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);
	virtual void backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);

	Dtype power_;
	Dtype scale_;
	Dtype shift_;
	Dtype diff_scale_;
};

# endif