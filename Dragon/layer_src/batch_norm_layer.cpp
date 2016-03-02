#include "layer_include/common_layers.hpp"
template <typename Dtype>
void BatchNormLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	BatchNormParameter norm_param = param.batch_norm_param();
	use_global_stats = norm_param.use_global_stats();
	eps = norm_param.eps();
	decay = norm_param.decay();
	//  conv layer norm on the basis of channels
	//	and fully-connected layer's feature axis is shape(1) also
	if (bottom[0]->num_axes() == 1) channels = 1;
	else channels = bottom[0]->shape(1);
	int batch_size = bottom[0]->shape(0);
	if (blobs.size() > 0){
		LOG(INFO) << "Checked previous params and skipped initialization";
	}else{
		vector<int> vec;
		vec.push_back(channels);
		//	beta
		blobs.push_back(boost::shared_ptr< Blob<Dtype> > (new Blob<Dtype>(vec)));
		//	gamma
		blobs.push_back(boost::shared_ptr< Blob<Dtype> >(new Blob<Dtype>(vec)));
		//	history mean
		blobs.push_back(boost::shared_ptr< Blob<Dtype> >(new Blob<Dtype>(vec)));
		//	history var
		blobs.push_back(boost::shared_ptr< Blob<Dtype> >(new Blob<Dtype>(vec)));
		//	use big precision help converging much faster 
		//	see https://github.com/ducha-aiki/caffe/blob/elu/examples/BN-nator.ipynb
		dragon_set<Dtype>(blobs[0]->count(), Dtype(0.0000001), blobs[0]->mutable_cpu_data());
		dragon_set<Dtype>(blobs[1]->count(), Dtype(1.0000001), blobs[1]->mutable_cpu_data());
		//	others will set to zero(done in syncedmem)
	} 
	param_need_bp.resize(2, true);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	check norm channels
	if (bottom[0]->num_axes() >= 1) CHECK_EQ(bottom[0]->shape(1), channels);
	top[0]->reshapeLike(*bottom[0]);
	vector<int> shape;
	shape.push_back(channels);
	mean.reshape(shape);
	var.reshape(shape);
	//	temp blob for storing
	temp.reshapeLike(*bottom[0]);
	expand_var.reshapeLike(*bottom[0]);
	x_norm.reshapeLike(*bottom[0]);
	//	set multiplier
	shape[0] = bottom[0]->shape(0);
	int batch_size = shape[0];
	batch_sum_multiplier.reshape(shape);
	int spatial_dim = bottom[0]->count() / (channels*batch_size);

	//	optimization for set op
	if (spatial_sum_multiplier.num_axes() == 0 ||
		spatial_sum_multiplier.shape(0) != spatial_dim){
		shape[0] = spatial_dim;
		spatial_sum_multiplier.reshape(shape);
		dragon_set(spatial_sum_multiplier.count(), Dtype(1), spatial_sum_multiplier.mutable_cpu_data());
	}
	int num_by_chans = channels*batch_size;
	if (num_by_channels.num_axes() == 0 ||
		num_by_channels.shape(0) != num_by_chans){
		shape[0] = num_by_chans;
		num_by_channels.reshape(shape);
		dragon_set(batch_sum_multiplier.count(), Dtype(1), batch_sum_multiplier.mutable_cpu_data());
	}
}


template <typename Dtype>
void BatchNormLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* beta_data = blobs[0]->mutable_cpu_data();
	Dtype* gamma_data = blobs[1]->mutable_cpu_data();
	int batch_size = bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (channels*batch_size);
	//	compute x^2 in temp
	dragon_powx<Dtype>(bottom[0]->count(), bottom_data, Dtype(2), temp.mutable_cpu_data());
	//	compute through all batches(sum each batch here)
	//	var[x]=E[x^2]-(E[x])^2
	//	sum all examples(batch_size*channels) for each dim(x)
	dragon_cpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
			Dtype(1) / (batch_size*spatial_dim), bottom_data, spatial_sum_multiplier.cpu_data(),
			Dtype(0), num_by_channels.mutable_cpu_data());
	//	sum the whole batch to get the mean
	dragon_cpu_gemv<Dtype>(CblasTrans, batch_size, channels,
			Dtype(1), num_by_channels.cpu_data(), batch_sum_multiplier.cpu_data(),
			Dtype(0), mean.mutable_cpu_data());
	//	sum all examples(batch_size*channels) for each dim(x^2)
	dragon_cpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
			Dtype(1) / (batch_size*spatial_dim), temp.cpu_data(), spatial_sum_multiplier.cpu_data(),
			Dtype(0), num_by_channels.mutable_cpu_data());
	//	sum the whole batch to get var(subtract after)
	dragon_cpu_gemv<Dtype>(CblasTrans, batch_size, channels,
			Dtype(1), num_by_channels.cpu_data(), batch_sum_multiplier.cpu_data(),
			Dtype(0), var.mutable_cpu_data());

	//	subtract and get the var
	dragon_powx<Dtype>(mean.count(), mean.cpu_data(), Dtype(2), temp.mutable_cpu_data());
	dragon_sub<Dtype>(var.count(), var.cpu_data(), temp.cpu_data(), var.mutable_cpu_data());
	//	add eps
	dragon_add_scalar(var.count(), eps, var.mutable_cpu_data());
	//	sqrt
	dragon_powx(var.count(), var.cpu_data(), Dtype(0.5), var.mutable_cpu_data());

	//	check if do not use in-place method
	//	we recommend use in-place like a activation function
	dragon_copy<Dtype>(bottom[0]->count(),top_data,bottom_data);

	//	subtract mean
	//	copy the mean for all examples in a batch
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.cpu_data(), mean.cpu_data(), Dtype(0),
		num_by_channels.mutable_cpu_data());
	//	copy and subtract
	//	if in conv layer, each feature map just subtract a mean value
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(-1), num_by_channels.cpu_data(), spatial_sum_multiplier.cpu_data(), Dtype(1),
		top_data);

	//	div std
	//	expand
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.cpu_data(), var.cpu_data(), Dtype(0),
		num_by_channels.mutable_cpu_data());
	//	expand
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.cpu_data(),spatial_sum_multiplier.cpu_data(),Dtype(0),
		temp.mutable_cpu_data());
	//	store sqrt(var+eps) for backward compution
	dragon_copy<Dtype>(expand_var.count(), expand_var.mutable_cpu_data(), temp.cpu_data());

	dragon_div(top[0]->count(), top_data, temp.cpu_data(), top_data);

	//	store x_norm for backward compution
	dragon_copy<Dtype>(x_norm.count(), x_norm.mutable_cpu_data(), top_data);

	//	scale gamma
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.cpu_data(), gamma_data, Dtype(0),
		num_by_channels.mutable_cpu_data());
	//	expand
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.cpu_data(), spatial_sum_multiplier.cpu_data(), Dtype(0),
		temp.mutable_cpu_data());
	dragon_mul(temp.count(), temp.cpu_data(), top_data,top_data);

	//	shift beta
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.cpu_data(), beta_data, Dtype(0),
		num_by_channels.mutable_cpu_data());
	//	copy2
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.cpu_data(), spatial_sum_multiplier.cpu_data(), Dtype(1),
		top_data);

}

template <typename Dtype>
void BatchNormLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	const Dtype *top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* gamma_data = blobs[1]->mutable_cpu_data();
	const Dtype* beta_data = blobs[0]->mutable_cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* beta_diff = blobs[0]->mutable_cpu_diff();
	Dtype* gamma_diff = blobs[1]->mutable_cpu_diff();
	int batch_size = bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (channels*batch_size);
	/*****    compute param_diff     *****/

	//	top_diff*x_norm
	dragon_mul(temp.count(), x_norm.cpu_data(), top_diff, temp.mutable_cpu_data());
	//	shrink by spatial_dim
	dragon_cpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
		Dtype(1), temp.cpu_data(), spatial_sum_multiplier.cpu_data(), Dtype(0),
		num_by_channels.mutable_cpu_data());
	//	shrink by batch_size and get gamma_diff
	dragon_cpu_gemv<Dtype>(CblasTrans, batch_size, channels,
		Dtype(1), num_by_channels.cpu_data(), batch_sum_multiplier.cpu_data(), Dtype(0), gamma_diff);

	//	shrink top_diff by spatial dim
	dragon_cpu_gemv<Dtype>(CblasNoTrans, batch_size*channels, spatial_dim,
		Dtype(1), top_diff, spatial_sum_multiplier.cpu_data(), Dtype(0),
		num_by_channels.mutable_cpu_data());
	//	shrink top_diff by batch_size and get beta_diff
	dragon_cpu_gemv<Dtype>(CblasTrans, batch_size, channels,
		Dtype(1), num_by_channels.cpu_data(), batch_sum_multiplier.cpu_data(), Dtype(0), beta_diff);


	/*****    compute x_norm_diff    *****/

	//	expand gamma by batch_size
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, channels, 1,
		Dtype(1), batch_sum_multiplier.cpu_data(), gamma_data, Dtype(0),
		num_by_channels.mutable_cpu_data());

	//	expand gamma by spatial_dim in temp
	dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size*channels, spatial_dim, 1,
		Dtype(1), num_by_channels.cpu_data(), spatial_sum_multiplier.cpu_data(), Dtype(0),
		temp.mutable_cpu_data());

	//	compute x_norm_diff
	dragon_mul<Dtype>(temp.count(), top_diff, temp.cpu_data(), x_norm.mutable_cpu_diff());

}

INSTANTIATE_CLASS(BatchNormLayer);