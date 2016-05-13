#include "layers/neuron/relu_layer.hpp"

template <typename Dtype>
__global__ void ReLU_forward_kernel(const int n, const Dtype* bottom_data, Dtype* top_data, Dtype slope){
	CUDA_KERNEL_LOOP(idx, n){
		top_data[idx] = bottom_data[idx] > 0 ? bottom_data[idx] : slope*bottom_data[idx];
	}
}

template <typename Dtype>
void ReLULayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int cnt = bottom[0]->count();
	Dtype slope = param.relu_param().negative_slope();
	ReLU_forward_kernel<Dtype> << < GET_BLOCKS(cnt), CUDA_NUM_THREADS >> >(
		cnt, bottom_data, top_data, slope);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReLU_backward_kernel(const int n, const Dtype* top_diff, const Dtype* bottom_data,
	Dtype* bottom_diff, Dtype slope){
	CUDA_KERNEL_LOOP(idx, n){
		bottom_diff[idx] = top_diff[idx] * ((bottom_data[idx] > 0)
			+ slope*(bottom_data[idx] <= 0));
	}
}

template <typename Dtype>
void ReLULayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (data_need_bp[0]){
		Dtype slope = param.relu_param().negative_slope();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int cnt = bottom[0]->count();
		//	bottom_diff = top_diff*1
		//				= slope
		ReLU_backward_kernel<Dtype> << < GET_BLOCKS(cnt), CUDA_NUM_THREADS >> >(
			cnt, top_diff, bottom_data, bottom_diff, slope);
	}
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);