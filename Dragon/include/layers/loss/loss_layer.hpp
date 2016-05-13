#ifndef LOSS_LAYER_HPP
#define LOSS_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class LossLayer :public Layer < Dtype > {
public:
	explicit LossLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
};

template <typename Dtype>
void LossLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	// auto add solve loss option if it is a LossLayer
	if (param.loss_weight_size() == 0) param.add_loss_weight(Dtype(1));
}

template <typename Dtype>
void LossLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
		<< "Data and Label should have same size.";
	//	loss just need an memory unit for storing
	vector<int> loss_shape(1, 1);
	top[0]->reshape(loss_shape);
}

# endif