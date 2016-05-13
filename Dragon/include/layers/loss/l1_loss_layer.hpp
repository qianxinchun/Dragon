# ifndef L1_LOSS_LAYER_HPP
# define L1_LOSS_LAYER_HPP

#include "loss_layer.hpp"

template <typename Dtype>
class SmoothL1LossLayer :public LossLayer < Dtype > {
public:
	SmoothL1LossLayer(const LayerParameter& param) :LossLayer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	Blob<Dtype> diff, errors, ones;
	bool has_weights;
	Dtype sigma2;
};

# endif
