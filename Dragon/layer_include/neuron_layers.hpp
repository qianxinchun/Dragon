#ifndef NEURON_LAYERS_HPP
#define NEURON_LAYERS_HPP
#include "layer_include/layer.hpp"
template <typename Dtype>
class NeuronLayer :public Layer < Dtype > {
public:
	NeuronLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
		top[0]->reshapeLike(*bottom[0]);
	}
};

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

template <typename Dtype>
class DropoutLayer :public NeuronLayer < Dtype > {
public:
	DropoutLayer(const LayerParameter& param) :NeuronLayer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	Dtype prob;
	unsigned int threshold;
	Blob<unsigned int> rand_vec;
};
#endif