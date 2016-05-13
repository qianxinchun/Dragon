# ifndef CONV_LAYER_HPP
# define CONV_LAYER_HPP

#include "base_conv_layer.hpp"

template<typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer < Dtype > {
public:
	ConvolutionLayer(const LayerParameter& param) :BaseConvolutionLayer<Dtype>(param) {}
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void computeOutputShape();
	virtual bool reverseDimensions() { return false; }
};

template<typename Dtype>
class DeconvolutionLayer : public BaseConvolutionLayer < Dtype > {
public:
	DeconvolutionLayer(const LayerParameter& param) :BaseConvolutionLayer<Dtype>(param) {}
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom);
	virtual void computeOutputShape();
	virtual bool reverseDimensions() { return true; }
};

# endif