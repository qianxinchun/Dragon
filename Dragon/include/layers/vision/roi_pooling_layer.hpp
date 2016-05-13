# ifndef ROI_POOLING_LAYER_HPP
# define ROI_POOLING_LAYER_HPP

#include "../../layer.hpp"

template <typename Dtype>
class ROIPoolingLayer :public Layer < Dtype > {
public:
	ROIPoolingLayer(const LayerParameter& param) :Layer(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	int pooling_height, pooling_width, height, width, channels;
	Dtype spatial_scale;
	Blob<int> max_idx;
};

# endif