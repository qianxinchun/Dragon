# ifndef ACCURACY_LAYER_HPP
# define ACCURACY_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class AccuracyLayer :public Layer < Dtype > {
public:
	AccuracyLayer(const LayerParameter& param) :Layer<Dtype>(param){}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	//	need not implement
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	int top_k, axis, outer_num, inner_num, ignore_label;
	bool has_ignore_label;
	Blob<Dtype> nums_buffer;

};

# endif
