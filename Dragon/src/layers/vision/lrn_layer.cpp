#include "layers/vision/lrn_layer.hpp"

template <typename Dtype>
void LRNLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LRNParameter lrn_param = param.lrn_param();
	size_ = lrn_param.local_size();
	CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
	pre_pad_ = (size_ - 1) / 2;
	alpha_ = lrn_param.alpha();
	beta_ = lrn_param.beta();
	k_ = lrn_param.k();
	if (lrn_param.norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL) {
		// Set up split_layer_ to use inputs in the numerator and denominator.
		split_top_vec_.clear();
		split_top_vec_.push_back(&product_input_);
		split_top_vec_.push_back(&square_input_);
		LayerParameter split_param;
		split_layer_.reset(new SplitLayer<Dtype>(split_param));
		split_layer_->setup(bottom, split_top_vec_);
		// Set up square_layer_ to square the inputs.
		square_bottom_vec_.clear();
		square_top_vec_.clear();
		square_bottom_vec_.push_back(&square_input_);
		square_top_vec_.push_back(&square_output_);
		LayerParameter square_param;
		square_param.mutable_power_param()->set_power(Dtype(2));
		square_layer_.reset(new PowerLayer<Dtype>(square_param));
		square_layer_->setup(square_bottom_vec_, square_top_vec_);
		// Set up pool_layer_ to sum over square neighborhoods of the input.
		pool_top_vec_.clear();
		pool_top_vec_.push_back(&pool_output_);
		LayerParameter pool_param;
		pool_param.mutable_pooling_param()->set_method(PoolingParameter_Method_AVG);
		pool_param.mutable_pooling_param()->set_pad(pre_pad_);
		pool_param.mutable_pooling_param()->set_kernel(size_);
		pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
		pool_layer_->setup(square_top_vec_, pool_top_vec_);
		// Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
		// the sum of a squared neighborhood (the output of pool_layer_).


		power_top_vec_.clear();
		power_top_vec_.push_back(&power_output_);
		LayerParameter power_param;
		power_param.mutable_power_param()->set_power(-beta_);
		power_param.mutable_power_param()->set_scale(alpha_);
		power_param.mutable_power_param()->set_shift(Dtype(1));
		power_layer_.reset(new PowerLayer<Dtype>(power_param));
		power_layer_->setup(pool_top_vec_, power_top_vec_);


		// Set up a product_layer_ to compute outputs by multiplying inputs by the
		// inverse demoninator computed by the power layer.
		product_bottom_vec_.clear();
		product_bottom_vec_.push_back(&product_input_);
		product_bottom_vec_.push_back(&power_output_);
		LayerParameter product_param;
		EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
		eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
		product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
		product_layer_->setup(product_bottom_vec_, top);
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	switch (param.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		top[0]->reshape(num_, channels_, height_, width_);
		scale_.reshape(num_, channels_, height_, width_);
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		split_layer_->reshape(bottom, split_top_vec_);
		square_layer_->reshape(square_bottom_vec_, square_top_vec_);
		pool_layer_->reshape(square_top_vec_, pool_top_vec_);
		power_layer_->reshape(pool_top_vec_, power_top_vec_);
		product_layer_->reshape(product_bottom_vec_, top);
		break;
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	switch (param.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		NOT_IMPLEMENTED;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		WithinChannelForward(bottom, top);
		break;
	default:
		LOG(FATAL) << "Unknown normalization region.";
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelForward(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	split_layer_->forward(bottom, split_top_vec_);
	square_layer_->forward(square_bottom_vec_, square_top_vec_);
	pool_layer_->forward(square_top_vec_, pool_top_vec_);
	power_layer_->forward(pool_top_vec_, power_top_vec_);
	product_layer_->forward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRNLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom) {
	switch (param.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		NOT_IMPLEMENTED;
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		WithinChannelBackward(top, data_need_bp, bottom);
		break;
	default:
		LOG(FATAL) << "Unknown normalization region.";
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(
	const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp,
	const vector<Blob<Dtype>*>& bottom) {
	if (data_need_bp[0]) {
		vector<bool> product_propagate_down(2, true);
		product_layer_->backward(top, product_propagate_down, product_bottom_vec_);
		power_layer_->backward(power_top_vec_, data_need_bp, pool_top_vec_);
		pool_layer_->backward(pool_top_vec_, data_need_bp, square_top_vec_);
		square_layer_->backward(square_top_vec_, data_need_bp, square_bottom_vec_);
		split_layer_->backward(split_top_vec_, data_need_bp, bottom);
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	switch (param.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		NOT_IMPLEMENTED;
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		WithinChannelForward(bottom, top);
		break;
	default:
		LOG(FATAL) << "Unknown normalization region.";
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& data_neep_bp, const vector<Blob<Dtype>*>& bottom) {
	switch (param.lrn_param().norm_region()) {
	case LRNParameter_NormRegion_ACROSS_CHANNELS:
		NOT_IMPLEMENTED;
		break;
	case LRNParameter_NormRegion_WITHIN_CHANNEL:
		WithinChannelBackward(top, data_neep_bp, bottom);
		break;
	default:
		LOG(FATAL) << "Unknown normalization region.";
	}
}

INSTANTIATE_CLASS(LRNLayer);