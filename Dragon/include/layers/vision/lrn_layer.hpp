# ifndef LRN_LAYER_HPP
# define LRN_LAYER_HPP

#include "../common/eltwise_layer.hpp"
#include "../common/split_layer.hpp"
#include "../vision/pooling_layer.hpp"
#include "../neuron/power_layer.hpp"

template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
public:
	LRNLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);
	virtual void backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);
	virtual void WithinChannelForward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void WithinChannelBackward(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);
	int size_;
	int pre_pad_;
	Dtype alpha_;
	Dtype beta_;
	Dtype k_;
	int num_;
	int channels_;
	int height_;
	int width_;

	// Fields used for normalization ACROSS_CHANNELS
	// scale_ stores the intermediate summing results
	Blob<Dtype> scale_;

	// Fields used for normalization WITHIN_CHANNEL
	boost::shared_ptr<SplitLayer<Dtype> > split_layer_;
	vector<Blob<Dtype>*> split_top_vec_;
	boost::shared_ptr<PowerLayer<Dtype> > square_layer_;
	Blob<Dtype> square_input_;
	Blob<Dtype> square_output_;
	vector<Blob<Dtype>*> square_bottom_vec_;
	vector<Blob<Dtype>*> square_top_vec_;
	boost::shared_ptr<PoolingLayer<Dtype> > pool_layer_;
	Blob<Dtype> pool_output_;
	vector<Blob<Dtype>*> pool_top_vec_;
	boost::shared_ptr<PowerLayer<Dtype> > power_layer_;
	Blob<Dtype> power_output_;
	vector<Blob<Dtype>*> power_top_vec_;
	boost::shared_ptr<EltwiseLayer<Dtype> > product_layer_;
	Blob<Dtype> product_input_;
	vector<Blob<Dtype>*> product_bottom_vec_;
};

# endif
