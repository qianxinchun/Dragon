#include "layer_include/vision_layers.hpp"
#include "alternative/device_alternative.hpp"

//	serial computions are splitted in parallel as 
//	num*channels*pooling_height*pooling_height units
//	it is a highly efficient parallel splitted algorithm
template<typename Dtype>
__global__ void MaxPoolForward(const int n, const Dtype* bottom_data, const int num, const int channels,
	const int height, const int width, const int pooling_height, const int pooling_width,
	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	Dtype* top_data, int* mask, Dtype* top_mask){
	CUDA_KERNEL_LOOP(idx, n){
		const int pw = idx%pooling_width;
		const int ph = (idx / pooling_width) % pooling_height;
		const int pc = (idx / pooling_width / pooling_height) % channels;
		const int pn = (idx / pooling_width / pooling_height / channels);
		int start_h = ph*stride_h - pad_h;
		int start_w = pw*stride_w - pad_w;
		//	clip
		const int end_h = min(start_h + kernel_h, height);
		const int end_w = min(start_w + kernel_w, width);
		start_h = max(start_h, 0);
		start_w = max(start_w, 0);
		Dtype max_val = -FLT_MAX;
		int max_idx = -1;
		//	base + offset(for num and channels)
		//	bottom_ptr pointer to a bottom map's base address
		const Dtype* bottom_ptr = bottom_data + (pn*channels + pc)*height*width;
		//	scan for the max val
		for (int h = start_h; h < end_h; h++){
			for (int w = start_w; w < end_w; w++){
				if (bottom_ptr[h*width+w] > max_val){
					max_idx = h*width + w;
					max_val = bottom_ptr[max_idx];
				}
			}
		}
		top_data[idx] = max_val;
		if (mask) mask[idx] = max_idx;
		else top_mask[idx] = max_idx;
	}
}

template<typename Dtype>
__global__ void AvgPoolForward(const int n, const Dtype* bottom_data, const int num, const int channels,
	const int height, const int width, const int pooling_height, const int pooling_width,
	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w,
	Dtype* top_data){
	CUDA_KERNEL_LOOP(idx, n){
		const int pw = idx%pooling_width;
		const int ph = (idx / pooling_width) % pooling_height;
		const int pc = (idx / pooling_width / pooling_height) % channels;
		const int pn = (idx / pooling_width / pooling_height / channels);
		int start_h = ph*stride_h - pad_h;
		int start_w = pw*stride_w - pad_w;
		int end_h = min(start_h + kernel_h, height + pad_h);
		int end_w = min(start_w + kernel_w, width + pad_w);
		const int pooling_size = (end_h - start_h)*(end_w - start_w);
		//	clip
		start_h = max(start_h, 0);
		start_w = max(start_w, 0);
		end_h = min(end_h, height);
		end_w = min(end_w, width);
		//	base + offset(for num and channels)
		//	bottom_ptr pointer to a bottom map's base address
		const Dtype* bottom_ptr = bottom_data + (pn*channels + pc)*height*width;
		Dtype avg_val = 0;
		//	scan for the max val
		for (int h = start_h; h < end_h; h++)
			for (int w = start_w; w < end_w; w++)
					avg_val+=bottom_ptr[h*width+w];
		top_data[idx] = avg_val / pooling_size;
	}
}

template<typename Dtype>
void PoolingLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	PoolingParameter pool_param = param.pooling_param();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int top_count = top[0]->count();
	const bool use_top_mask = top.size() > 1;
	int *mask = NULL;
	Dtype *top_mask = NULL;
	switch (pool_param.method()){
	case PoolingParameter_Method_MAX:
		if (use_top_mask) top_mask = top[1]->mutable_gpu_data();
		else mask = max_idx.mutable_gpu_data();
		MaxPoolForward<Dtype> << <GET_BLOCKS(top_count), CUDA_NUM_THREADS >> >(
			top_count, bottom_data, bottom[0]->num(), channels, height, width,
			pooling_height, pooling_width, kernel_h, kernel_w, stride_h, stride_w,
			pad_h, pad_w, top_data, mask, top_mask);
		break;
	case PoolingParameter_Method_AVG:
		AvgPoolForward<Dtype> << <GET_BLOCKS(top_count), CUDA_NUM_THREADS >> >(
			top_count, bottom_data, bottom[0]->num(), channels, height, width,
			pooling_height, pooling_width, kernel_h, kernel_w, stride_h, stride_w,
			pad_h, pad_w, top_data);
		break;
	case PoolingParameter_Method_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
	CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
__global__ void MaxPoolBackward(const int n, const Dtype* top_diff, const int num, const int channels,
	const int height, const int width, const int pooling_height, const int pooling_width,
	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
	const int pad_h, const int pad_w, Dtype* bottom_diff, const int* mask, const Dtype* top_mask){
	CUDA_KERNEL_LOOP(idx, n){
		const int w = idx%width;
		const int h = (idx / width) % height;
		const int c = (idx / width / height) % channels;
		const int n = idx / width / height / channels;
		//	allow overlapping
		const int start_ph = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
		const int start_pw = (w + pad_w<kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
		//	allow clip
		const int end_ph = min((h + pad_h) / stride_h + 1, pooling_height);
		const int end_pw = min((w + pad_w) / stride_w + 1, pooling_width);
		Dtype diff = 0;
		const int offset = (n*channels + c)*pooling_height*pooling_width;
		const Dtype* top_ptr = top_diff + offset;
		if (mask){
			const int* mask_ptr = mask + offset;
			for (int ph = start_ph; ph < end_ph; ph++)
				for (int pw = start_pw; pw < end_pw; pw++)
					if (mask_ptr[ph*pooling_width + pw] == (h*width + w))
						diff += top_ptr[ph*pooling_width + pw];
		}else{
			const Dtype* mask_ptr = top_mask + offset;
			for (int ph = start_ph; ph < end_ph; ph++)
				for (int pw = start_pw; pw < end_pw; pw++)
					if (mask_ptr[ph*pooling_width + pw] == (h*width + w))
						diff += top_ptr[ph*pooling_width + pw];
		}
		bottom_diff[idx] = diff;
	}
}

template<typename Dtype>
__global__ void AvgPoolBackward(const int n, const Dtype* top_diff, const int num, const int channels,
	const int height, const int width, const int pooling_height, const int pooling_width,
	const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
	const int pad_h, const int pad_w, Dtype* bottom_diff){
	CUDA_KERNEL_LOOP(idx, n){
		const int w = idx%width;
		const int h = (idx / width) % height;
		const int c = (idx / width / height) % channels;
		const int n = idx / width / height / channels;
		//	allow overlapping
		const int start_ph = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
		const int start_pw = (w + pad_w<kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
		//	allow clip
		//	note that use 'h / stride_h + 1' but not '(h + pad_h) / stride_h + 1'
		//	will ignore pad when average(???)
		const int end_ph = min(h / stride_h + 1, pooling_height);
		const int end_pw = min(w / stride_w + 1, pooling_width);
		Dtype diff = 0;
		const Dtype* top_ptr = top_diff+(n*channels + c)*pooling_height*pooling_width;
		for (int ph = start_ph; ph < end_ph; ph++)
			for (int pw = start_pw; pw < end_pw; pw++){
			//	must compute pooling size
			int start_h = ph*stride_h - pad_h;
			int start_w = pw*stride_w - pad_w;
			int end_h = min(start_h + kernel_h, height + pad_h);
			int end_w = min(start_w + kernel_w, width + pad_w); 
			int pooling_size = (end_h - start_h)*(end_w - start_w);
			diff += (top_ptr[ph*pooling_width + pw] / pooling_size);
		}
		bottom_diff[idx] = diff;
	}
}

template<typename Dtype>
void PoolingLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	// pooling layer only compute data_diff
	if (!data_need_bp[0]) return;
	PoolingParameter pool_param = param.pooling_param();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int bottom_count = bottom[0]->count();
	const bool use_top_mask = top.size() > 1;
	const int* mask = NULL;
	const Dtype* top_mask = NULL;
	switch (pool_param.method()){
	case PoolingParameter_Method_MAX:
		if (use_top_mask) top_mask = top[1]->gpu_data();
		else mask = max_idx.gpu_data();
		MaxPoolBackward<Dtype> << <GET_BLOCKS(bottom_count), CUDA_NUM_THREADS >> >(
			bottom_count, top_diff, bottom[0]->num(), channels, height, width,
			pooling_height, pooling_width, kernel_h, kernel_w, stride_h, stride_w,
			pad_h, pad_w, bottom_diff, mask, top_mask);
		break;
	case PoolingParameter_Method_AVG:
		AvgPoolBackward<Dtype> << <GET_BLOCKS(bottom_count), CUDA_NUM_THREADS >> >(
			bottom_count, top_diff, bottom[0]->num(), channels, height, width,
			pooling_height, pooling_width, kernel_h, kernel_w, stride_h, stride_w,
			pad_h, pad_w, bottom_diff);
		break;
	case PoolingParameter_Method_STOCHASTIC:
		NOT_IMPLEMENTED;
		break;
	default:
		LOG(FATAL) << "Unknown pooling method.";
	}
	CUDA_POST_KERNEL_CHECK; 
}

INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);