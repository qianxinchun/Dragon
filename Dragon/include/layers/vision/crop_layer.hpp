# ifndef CROP_LAYER_HPP
# define CROP_LAYER_HPP

#include "../../layer.hpp"

#define LEVEL 3

template <typename Dtype>
class CropLayer :public Layer < Dtype > {
public:
	CropLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
private:
	void copy(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top,
		const vector<int>& offsets, vector<int> idxs, int cur_dim, const Dtype* src_data,
		Dtype* dest_data, bool is_forward);
	void copy_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top,
		const vector<int>& offsets, vector<int> idxs, int cur_dim, const Dtype* src_data,
		Dtype* dest_data, bool is_forward);
	vector<int> offsets;
	Blob<int> src_dims_blob, dest_dims_blob;
};


# endif