# ifndef ELTWISE_LAYER_HPP
# define ELTWISE_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> {
public:
	EltwiseLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);
	virtual void backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);

	EltwiseParameter_EltwiseOp op_;
	vector<Dtype> coeffs_;
	Blob<int> max_idx_;

	bool stable_prod_grad_;
};

# endif