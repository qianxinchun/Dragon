#ifndef LSTM_LAYER_HPP
#define LSTM_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class LSTMLayer :public Layer < Dtype > {
public:
	LSTMLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {}
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	int input_dim, hidden_dim;
	int batch_size, steps;
	Dtype clipping_threshold;
	Blob<Dtype> bias_multiplier;
	Blob<Dtype> output, cell, pre_gate, gate;
	Blob<Dtype> c_1, h_1, c_T, h_T;
	Blob<Dtype> h_to_gate, h_to_h;
};


#endif