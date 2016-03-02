#include "layer_include/vision_layers.hpp"
#include "include/blob.hpp"
#include "cmath"
template<typename Dtype>
void PoolingLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	PoolingParameter pool_param = param.pooling_param();
	//	user need not set the kernel size if use global pooling
	if (pool_param.global_pooling()){
		CHECK(!(pool_param.has_kernel() ||
			pool_param.has_kernel_h() || pool_param.has_kernel_w()))
			<< "Can not specific kernel when using global pooling.";
	}
	else{
		if (pool_param.has_kernel() && (pool_param.has_kernel_w() && pool_param.has_kernel_w()))
			LOG(FATAL) << "Can not specific kernel or kernel_h/w both at the same time.";
		if (!pool_param.has_kernel() && !(pool_param.has_kernel_w() && pool_param.has_kernel_w()))
			LOG(FATAL) << "Must specific kernel or kernel_h/w both either.";
	}
	if (pool_param.has_pad() && (pool_param.has_kernel_w() && pool_param.has_kernel_w()))
		LOG(FATAL) << "Can not specific kernel or kernel_h/w both at the same time.";
	if (!pool_param.has_kernel() && !(pool_param.has_kernel_w() && pool_param.has_kernel_w()))
		LOG(FATAL) << "Must specific kernel or kernel_h/w both either.";
	global_pooling = pool_param.global_pooling();

	//	set kernel
	if (global_pooling){
		kernel_h = bottom[0]->height();
		kernel_w = bottom[0]->width();
	}
	else{
		if (pool_param.has_kernel()) kernel_h = kernel_w = pool_param.kernel();
		else{
			kernel_h = pool_param.kernel_h();
			kernel_w = pool_param.kernel_w();
		}
	}
	CHECK_GT(kernel_h, 0) << "kernel_h must greater than zero";
	CHECK_GT(kernel_w, 0) << "kernel_w must greater than zero";
	//	set pad
	if (pool_param.has_pad()) pad_h = pad_w = pool_param.pad();
	else{
		pad_h = pool_param.pad_h();
		pad_w = pool_param.pad_w();
	}
	//	set stride
	if (pool_param.has_stride()) stride_h = stride_w = pool_param.stride();
	else{
		stride_h = pool_param.stride_h();
		stride_w = pool_param.stride_w();
	}
	//	check pad/stride
	if (global_pooling)
		CHECK(pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1)
		<< "pad/stride must equal to zero/one when using global pooling.";
	//	check pad/kernel
	if (pad_h != 0 || pad_w != 0){
		CHECK(pool_param.method() == PoolingParameter_Method_MAX ||
			pool_param.method() == PoolingParameter_Method_AVG)
			<< "Padding only for MAX/AVG pooling.";
		CHECK_LT(pad_h, kernel_h) << "pad_h must less than kernel_h or do nothing.";
		CHECK_LT(pad_w, kernel_w) << "pad_w must less than kernel_w or do nothing.";
	}
}

template<typename Dtype>
void PoolingLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	PoolingParameter pool_param = param.pooling_param();
	CHECK_EQ(bottom[0]->num_axes(), 4) << "Input must be 4D blobs.";
	channels = bottom[0]->channels();
	height = bottom[0]->height();
	width = bottom[0]->width();
	// allow non-aligned pooling and add a unit separately if necessary
	pooling_height = ceil((height + 2 * pad_h - kernel_h) / (float)stride_h)+1;
	pooling_width = ceil((width + 2 * pad_w - kernel_w) / (float)stride_w) + 1;
	if (pad_h || pad_w){
		//	we ensure that the addtional pooling unit must be vaild
		//	it means that the last pooling area should start inside the image
		//	or this unit d o nothing and we clip the last pooling unit
		//	remember that the first pooling is vaild absolutely cause pad<kernel
		if ((pooling_height - 1)*stride_h >= (height + pad_h)) pooling_height--;
		if ((pooling_width - 1)*stride_w >= (width + pad_w)) pooling_width--;
	}
	top[0]->reshape(bottom[0]->num(), channels, pooling_width, pooling_height);
	//	use it for top_mask
	//	top_mask will replace the default max_idx blob when logging the pooling idx
	if (top.size() > 1) top[1]->reshapeLike(*top[0]); 
	//	do not use reshapeLike, cause top[0] is Dtype but max_idx is int
	if (pool_param.method() == PoolingParameter_Method_MAX&&top.size() == 1)
		max_idx.reshape(bottom[0]->num(), channels, pooling_height, pooling_width);
	if (pool_param.method() == PoolingParameter_Method_STOCHASTIC)
		rand_idx.reshapeLike(*top[0]);
}

template<typename Dtype>
void PoolingLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	PoolingParameter pool_param = param.pooling_param();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int top_count = top[0]->count();
	const bool use_top_mask = top.size() > 1;
	int *mask = NULL;
	Dtype *top_mask = NULL;
	switch (pool_param.method()){
	case PoolingParameter_Method_MAX:
		if (use_top_mask) top_mask = top[1]->mutable_cpu_data();
		else mask = max_idx.mutable_cpu_data();
		for (int n = 0; n < bottom[0]->num(); n++){
			for (int c = 0; c < channels; c++){
				for (int ph = 0; ph < pooling_height; ph++){
					for(int pw = 0; pw < pooling_width; pw++){
						//	compute the start position
						int start_h = ph*stride_h - pad_h;
						int start_w = pw*stride_w - pad_w;
						//	compute the end position
						//	clip the position due to padding at the end
						int end_h = min(start_h + kernel_h, height);
						int end_w = min(start_w + kernel_w, width);
						//	clip the position due to padding at the start
						start_h = max(start_h, 0);
						start_w = max(start_w, 0);
						//	pool_idx represents the x_th output unit
						const int pool_idx = ph*pooling_width + pw;
						//	for a fixed data and channel
						//	we scan the max val and log the idx for diff_computing
						//	note that bottom/top_data will offset later
						Dtype max_val = -FLT_MAX;
						int max_idx = -1;
						for (int h = start_h; h < end_h; h++){
							for (int w = start_w; w < end_w; w++){
								//	idx represents the y_th im unit which the x_th output unit used
								const int idx = h*width + w;
								if (bottom_data[idx]>max_val){
									max_val = bottom_data[idx];
									max_idx = idx;
								}
							}	//	end w
						}	//	end h
						top_data[pool_idx] = max_val;
						if (use_top_mask) top_mask[pool_idx] = max_idx;
						else mask[pool_idx] = max_idx;
					}	//	end pw
				}	//	end ph
				//	offset a channel
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
				if (use_top_mask) top_mask += top[0]->offset(0, 1);
				else mask += top[0]->offset(0, 1);
			}	//	end c
		}	//	end n
		break;

	case PoolingParameter_Method_AVG:
		dragon_set(top_count, Dtype(0), top_data);
		for (int n = 0; n < bottom[0]->num(); n++){
			for (int c = 0; c < channels; c++){
				for (int ph = 0; ph < pooling_height; ph++){
					for (int pw = 0; pw < pooling_width; pw++){
						int start_h = ph*stride_h - pad_h;
						int start_w = pw*stride_w - pad_w;
						int end_h = min(start_h + kernel_h, height + pad_h);
						int end_w = min(start_w + kernel_w, width + pad_w);
						//	before cilp we need compute the pool area for average
						int pool_area = (end_h - start_h)*(end_w - start_w);
						//	clip
						end_h = min(end_h, height);
						end_w = min(end_w, width);
						start_h = max(start_h, 0);
						start_w = max(start_w, 0);
						const int pool_idx = ph*pooling_width + pw;
						//	sum up all units in the area
						for (int h = start_h; h < end_h; h++){
							for (int w = start_w; w < end_w; w++){
								const int idx = h*width + w;
								top_data[pool_idx] += bottom_data[idx];
							}
						}
						//	do average
						top_data[pool_idx] /= pool_area; 
						//	note that AVG pooling need not log the idx for diff_computing
					}	//end pw
				}	//end ph
				bottom_data += bottom[0]->offset(0, 1);
				top_data += top[0]->offset(0, 1);
			}	//end c
		}	//end n
		break;

	case PoolingParameter_Method_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;

	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}

template<typename Dtype>
void PoolingLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	// pooling layer only compute data_diff
	if (!data_need_bp[0]) return;
	PoolingParameter pool_param = param.pooling_param();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	dragon_set(bottom[0]->count(), Dtype(0), bottom_diff);
	const bool use_top_mask = top.size() > 1;
	const int* mask = NULL;
	const Dtype* top_mask = NULL;
	switch (pool_param.method()){
	case PoolingParameter_Method_MAX:
		if (use_top_mask) top_mask = top[1]->cpu_data();
		else mask = max_idx.cpu_data();
		for (int n = 0; n < bottom[0]->num(); n++){
			for (int c = 0; c < channels; c++){
				for (int ph = 0; ph < pooling_height; ph++){
					for (int pw = 0; pw < pooling_width; pw++){
						const int pool_idx = ph*pooling_width + pw;
						const int idx = use_top_mask ? top_mask[pool_idx] : mask[pool_idx];
						//	bottom_diff += delta_(layer+1)
						//	note that we allow overlapping pooling
						//	it means that different top_diffs may have a same bottom_diff
						//	because bottom_diff may overlap
						//	use '+=' replace '=' if using overlapping pooling
						//	also, using idx can consider as to decide a contributed bottom_diff
						//	backward the sub gradient only to the contributed bottom_diff
						//	non-contributed bottom_diff will keep zero which is setted in dragon_set()
						bottom_diff[idx] += top_diff[pool_idx];
					}	//	end pw
				}//	end ph
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
				if (use_top_mask) top_mask += top[0]->offset(0, 1);
				else mask += top[0]->offset(0, 1);
			}	// end c
		}//	end n
		break;

	case PoolingParameter_Method_AVG:
		for (int n = 0; n < bottom[0]->num(); n++){
			for (int c = 0; c < channels; c++){
				for (int ph = 0; ph < pooling_height; ph++){
					for (int pw = 0; pw < pooling_width; pw++){
						int start_h = ph*stride_h - pad_h;
						int start_w = pw*stride_w - pad_w;
						int end_h = min(start_h + kernel_h, height + pad_h);
						int end_w = min(start_w + kernel_w, width + pad_w);
						//	before cilp we need compute the pool area for average
						int pool_area = (end_h - start_h)*(end_w - start_w);
						//	clip
						end_h = min(end_h, height);
						end_w = min(end_w + kernel_w, width);
						start_h = max(start_h, 0);
						start_w = max(start_w, 0);
						const int pool_idx = ph*pooling_width + pw;
						//	1/(pool_area)*bottom_data=top_data
						//  d(top_data)/d(bottom_data)=1/(pool_area)
						//	combine with sub gradient and we get 'top_diff[pool_idx] / pool_area'
						for (int h = start_h; h < end_h; h++){
							for (int w = start_w; w < end_w; w++){
								const int idx = h*width + w;
								bottom_diff[idx] += (top_diff[pool_idx] / pool_area);
							}
						}
					}	//	end pw
				}//	end ph
				bottom_diff += bottom[0]->offset(0, 1);
				top_diff += top[0]->offset(0, 1);
			}	// end c
		}//	end n
		break;

	case PoolingParameter_Method_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
}

INSTANTIATE_CLASS(PoolingLayer);