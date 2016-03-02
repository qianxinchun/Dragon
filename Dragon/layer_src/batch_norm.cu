#include "layer_include/common_layers.hpp"

template <typename Dtype>
void BatchNormLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype* bottom_data = bottom[0]->mutable_gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* beta_data = blobs[0]->mutable_gpu_data();
	Dtype* gamma_data = blobs[1]->mutable_gpu_data();
	int batch_size = bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (channels*batch_size);
	Dtype* history_mean = blobs[2]->mutable_gpu_data();
	Dtype* history_var = blobs[3]->mutable_gpu_data();
	//	copy bottom data
	dragon_gpu_copy<Dtype>(bottom[0]->count(), top_data, bottom_data);

	// compute mean and var for this batch
	if (phase == TRAIN||!use_global_stats){

		//	compute x^2 in temp
		dragon_gpu_powx<Dtype>(bottom[0]->count(), bottom_data, Dtype(2), temp.mutable_gpu_data());
		//	compute through all batches(sum each batch here)
		//	var[x]=E[x^2]-(E[x])^2
		//	sum all examples(batch_size*channels) for each dim(x)
		dragon_gpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
			Dtype(1) / (batch_size*spatial_dim), bottom_data, spatial_sum_multiplier.gpu_data(),
			Dtype(0), num_by_channels.mutable_gpu_data());
		//	sum the whole batch to get the mean
		dragon_gpu_gemv<Dtype>(CblasTrans, batch_size, channels,
			Dtype(1), num_by_channels.gpu_data(), batch_sum_multiplier.gpu_data(),
			Dtype(0), mean.mutable_gpu_data());
		//	sum all examples(batch_size*channels) for each dim(x^2)
		dragon_gpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
			Dtype(1) / (batch_size*spatial_dim), temp.gpu_data(), spatial_sum_multiplier.gpu_data(),
			Dtype(0), num_by_channels.mutable_gpu_data());
		//	sum the whole batch to get var(subtract after)
		dragon_gpu_gemv<Dtype>(CblasTrans, batch_size, channels,
			Dtype(1), num_by_channels.gpu_data(), batch_sum_multiplier.gpu_data(),
			Dtype(0), var.mutable_gpu_data());

		//	subtract and get the var
		dragon_gpu_powx<Dtype>(mean.count(), mean.gpu_data(), Dtype(2), temp.mutable_gpu_data());
		dragon_gpu_sub<Dtype>(var.count(), var.gpu_data(), temp.gpu_data(), var.mutable_gpu_data());
		//	add eps
		dragon_gpu_add_scalar(var.count(), eps, var.mutable_gpu_data());
		//	sqrt
		dragon_gpu_powx(var.count(), var.gpu_data(), Dtype(0.5), var.mutable_gpu_data());

		//	store history mean and var
		//	history=decay(0.95)*val+(1-decay)*histoty
		dragon_gpu_axpby(mean.count(), decay, mean.gpu_data(), Dtype(1)-decay, history_mean);
		dragon_gpu_axpby(var.count(), decay, var.gpu_data(), Dtype(1) - decay, history_var);

	}
	else if (phase == TEST&&use_global_stats){
		// copy the history data
		dragon_gpu_copy(mean.count(), mean.mutable_gpu_data(), history_mean);
		dragon_gpu_copy(var.count(), var.mutable_gpu_data(), history_var);
	}

	//	subtract mean
	//	copy the mean for all examples in a batch
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), mean.gpu_data(), Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	copy and subtract
	//	if in conv layer, each feature map just subtract a mean value
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(-1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(1),
		top_data);

	//	div std
	//	expand
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), var.gpu_data(), Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	expand
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(0),
		temp.mutable_gpu_data());

	//	store sqrt(var+eps) for backward compution
	dragon_gpu_copy<Dtype>(expand_var.count(), expand_var.mutable_gpu_data(), temp.gpu_data());

	dragon_gpu_div(top[0]->count(), top_data, temp.gpu_data(), top_data);

	//	store x_norm for backward compution
	dragon_gpu_copy<Dtype>(x_norm.count(), x_norm.mutable_gpu_data(), top_data);

	//	scale gamma
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), gamma_data, Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	expand
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(0),
		temp.mutable_gpu_data());
	dragon_gpu_mul(temp.count(), temp.gpu_data(), top_data, top_data);

	//	shift beta
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), beta_data, Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	copy2
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(1),
		top_data);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	const Dtype *top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* gamma_data = blobs[1]->mutable_gpu_data();
	const Dtype* beta_data = blobs[0]->mutable_gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* beta_diff = blobs[0]->mutable_gpu_diff();
	Dtype* gamma_diff = blobs[1]->mutable_gpu_diff();
	int batch_size = bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (channels*batch_size);
	/*****    compute param_diff     *****/

	//	top_diff*x_norm in temp.data
	dragon_gpu_mul(temp.count(), x_norm.gpu_data(), top_diff, temp.mutable_gpu_data());
	//	shrink by spatial_dim
	dragon_gpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
		Dtype(1), temp.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	shrink by batch_size and get gamma_diff
	dragon_gpu_gemv<Dtype>(CblasTrans, batch_size, channels,
		Dtype(1), num_by_channels.gpu_data(), batch_sum_multiplier.gpu_data(), Dtype(0), gamma_diff);

	//	shrink top_diff by spatial dim
	dragon_gpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
		Dtype(1), top_diff, spatial_sum_multiplier.gpu_data(), Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	shrink top_diff by batch_size and get beta_diff
	dragon_gpu_gemv<Dtype>(CblasTrans, batch_size, channels,
		Dtype(1), num_by_channels.gpu_data(), batch_sum_multiplier.gpu_data(), Dtype(0), beta_diff);


	/*****    compute x_norm_diff    *****/

	//	expand gamma by batch_size
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), gamma_data, Dtype(0),
		num_by_channels.mutable_gpu_data());

	//	expand gamma by spatial_dim in temp
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(0),
		temp.mutable_gpu_data());

	//	compute x_norm.diff
	dragon_gpu_mul<Dtype>(temp.count(), top_diff, temp.gpu_data(), x_norm.mutable_gpu_diff());

	//  compute x_norm.diff*[(x-mu)/sqrt(...)]=x_norm.diff*x_norm.data
	dragon_gpu_mul<Dtype>(x_norm.count(), x_norm.gpu_diff(), x_norm.gpu_data(), bottom_diff);

	//	compute var.diff
	//	shrink by spatial_dim
	dragon_gpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
		Dtype(1), bottom_diff, spatial_sum_multiplier.gpu_data(), Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	shrink by batch_size in var.diff
	dragon_gpu_gemv<Dtype>(CblasTrans, batch_size, channels,
		Dtype(1), num_by_channels.gpu_data(), batch_sum_multiplier.gpu_data(), Dtype(0), 
		var.mutable_gpu_diff());

	//	expand by batch_size
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), var.gpu_diff(), Dtype(0),
		num_by_channels.mutable_gpu_data());

	//	expand by spatial_dim in bottom_diff
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(0),
		bottom_diff);

	//	compute bottom_diff*[(x-mu)/sqrt(...)]=bottom_diff*x_norm.data in bottom_diff
	dragon_gpu_mul<Dtype>(x_norm.count(), bottom_diff, x_norm.gpu_data(), bottom_diff);

	//  compute mean.diff
	//	shrink x_norm.diff by spatial_dim
	dragon_gpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
		Dtype(1), x_norm.gpu_diff(), spatial_sum_multiplier.gpu_data(), Dtype(0),
		num_by_channels.mutable_gpu_data());
	//	shrink by batch_size in mean.diff
	dragon_gpu_gemv<Dtype>(CblasTrans, batch_size, channels,
		Dtype(1), num_by_channels.gpu_data(), batch_sum_multiplier.gpu_data(), Dtype(0),
		mean.mutable_gpu_diff());

	//	expand by batch_size
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.gpu_data(), mean.gpu_diff(), Dtype(0),
		num_by_channels.mutable_gpu_data());

	//	expand by spatial_dim and plus in bottom_diff
	dragon_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.gpu_data(), spatial_sum_multiplier.gpu_data(), Dtype(1),
		bottom_diff);

	//	m=batch_size*spatial_dim
	Dtype m = Dtype(bottom[0]->count() / channels);

	//	x_norm.diff-(1/m)*bottom_diff
	dragon_gpu_axpby(bottom[0]->count(), Dtype(1), x_norm.gpu_diff(), Dtype(-1.0 /m), bottom_diff);

	//	div sqrt(...)
	dragon_gpu_div(bottom[0]->count(), bottom_diff, expand_var.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);