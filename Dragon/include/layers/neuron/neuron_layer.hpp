#ifndef NEURON_LAYER_HPP
#define NEURON_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class NeuronLayer :public Layer < Dtype > {
public:
	NeuronLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
		top[0]->reshapeLike(*bottom[0]);
	}
};

# endif