#include "layers/common/concat_layer.hpp"

template <typename Dtype>
//	cuda kernel function run faster than cudaMemcpy
__global__ void Forward(int count, const Dtype* bottom_data, const int batch_size, const int input_size,
	const int num_output, const int channels, const int channels_offset, Dtype* top_data){
	CUDA_KERNEL_LOOP(idx, count){
		//	count ~ batch_size*channels*input_size
		//	idx ~ bottom_idx 
		//  the target is to solve top_idx
		int n = idx / input_size / channels;
		//	solve the idx in each example
		int copy_idx = idx % (input_size*channels);
		int top_idx = (n*num_output + channels_offset)*input_size + copy_idx;
		top_data[top_idx] = bottom_data[idx];
	}

}

template <typename Dtype>
void ConcatLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	if (bottom.size() == 1) return;
	Dtype* top_data = top[0]->mutable_gpu_data();
	int channels_offset = 0, num_output = top[0]->shape(axis);
	for (int i = 0; i < bottom.size(); i++){
		const Dtype* bottom_data = bottom[i]->gpu_data();
		const int channels = bottom[i]->shape(axis);
		int count = batch_size*channels*input_size;
		Forward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(count, bottom_data, batch_size,
			input_size, num_output, channels, channels_offset, top_data);
		channels_offset += channels;
	}
}

template <typename Dtype>
__global__ void Backward(int count, const Dtype* top_diff, const int batch_size, const int input_size,
	const int num_output, const int channels, const int channels_offset, Dtype* bottom_diff){
	CUDA_KERNEL_LOOP(idx, count){
		int n = idx / input_size / channels;
		int copy_idx = idx % (input_size*channels);
		int top_idx = (n*num_output + channels_offset)*input_size + copy_idx;
		bottom_diff[idx] = top_diff[top_idx];
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	if (bottom.size() == 1) return;
	const Dtype* top_diff = top[0]->gpu_diff();
	int channels_offset = 0, num_output = top[0]->shape(axis);
	for (int i = 0; i < bottom.size(); i++){
		const int channels = bottom[i]->shape(axis);
		if (data_need_bp[i]){
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			int count = batch_size*channels*input_size;
			Backward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(count, top_diff, batch_size,
				input_size, num_output, channels, channels_offset, bottom_diff);
		}
		channels_offset += channels;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);