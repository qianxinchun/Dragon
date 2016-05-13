# ifndef DATA_LAYER_HPP
# define DATA_LAYER_HPP

#include "prefetching_data_layer.hpp"

template<typename Dtype>
class DataLayer :public BasePrefetchingDataLayer < Dtype > {
public:
	DataLayer(const LayerParameter& param) :BasePrefetchingDataLayer(param) {}
	void dataLayerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>&top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
};

# endif
