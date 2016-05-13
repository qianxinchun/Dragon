#include "layers/vision/crop_layer.hpp"

//	a linear-mem copy kernel for the last two spatial axis
//	we re-implement it which is much efficient that caffe-master version
template <typename Dtype>
__global__ void	CopyKernel(const int n, const int height, const int width,
	const int src_outer_stride, const int src_inner_stride, const int dest_outer_stride, const int dest_inner_stride,
	const Dtype* src, Dtype* dest){
	CUDA_KERNEL_LOOP(idx, n){
		/*
		int src_start = idx / height*src_outer_stride
			+ idx%height*src_inner_stride;
		int dest_start = idx / height*dest_outer_stride
			+ idx%height*dest_inner_stride;*/
		int w = idx%width;
		int h = (idx / width) % height;
		int dest_idx = h*dest_inner_stride + w;
		int src_idx = h*src_inner_stride + w;
		dest[dest_idx] = src[src_idx];
	}
}


template <typename Dtype>
void CropLayer<Dtype>::copy_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top,
	const vector<int>& offsets, vector<int> idxs, int cur_dim, const Dtype* src_data,
	Dtype* dest_data, bool is_forward){

	//	recursive-term
	if (cur_dim + 2 < top[0]->num_axes()){
		for (int i = 0; i < top[0]->shape(cur_dim); i++){
			//	store the pixel-idx of the current spatial axis
			idxs[cur_dim] = i;
			//	recursive for spatial axis
			copy_gpu(bottom, top, offsets, idxs, cur_dim + 1, src_data, dest_data, is_forward);
		}
	}
	//	terminal-term
	//	perform a linear-mem copy for the last two spatial axis
	//	you can also perform a last-n parallel algorithms for cuda kernel function
	else{
		const int lines = top[0]->shape(cur_dim);
		const int height = top[0]->shape(cur_dim);
		const int width= top[0]->shape(cur_dim+1);
		const int outer_num = height*width;
		vector<int> idx_off(cur_dim + 2, 0);
		for (int j = 0; j < cur_dim; j++) idx_off[j] = idxs[j] + offsets[j];
		idx_off[cur_dim] = offsets[cur_dim];
		idx_off[cur_dim+1] = offsets[cur_dim+1];
		const int src_outer_stride =
			bottom[0]->shape(cur_dim)*bottom[0]->shape(cur_dim + 1);
		const int src_inner_stride = bottom[0]->shape(cur_dim + 1);
		const int dest_outer_stride =
			top[0]->shape(cur_dim)*top[0]->shape(cur_dim + 1);
		const int dest_inner_stride = top[0]->shape(cur_dim + 1);
		//
		if (is_forward){
			const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(idx_off);
			Dtype* top_data = top[0]->mutable_gpu_data() + top[0]->offset(idxs);
			CopyKernel<Dtype> << <GET_BLOCKS(outer_num), CUDA_NUM_THREADS >> >(
				outer_num, height, width, src_outer_stride, src_inner_stride,
				dest_outer_stride, dest_inner_stride,bottom_data, top_data);
		}else{
			const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(idxs);
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(idx_off);
			CopyKernel<Dtype> << <GET_BLOCKS(outer_num), CUDA_NUM_THREADS >> >(
				outer_num, height, width, dest_outer_stride, dest_inner_stride,
				src_outer_stride, src_inner_stride, top_diff, bottom_diff);
		}
	}
}

template <typename Dtype>
void CropLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	vector<int> idxs(top[0]->num_axes(), 0);
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	copy_gpu(bottom, top, offsets, idxs, 0, bottom_data, top_data, true);
}

template <typename Dtype>
void CropLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (!data_need_bp[0]) return;
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	//	must clear the last diff due to the different shape according mini-batches
	dragon_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
	vector<int> idxs(top[0]->num_axes(), 0);
	copy_gpu(bottom, top, offsets, idxs, 0, top_diff, bottom_diff, false);
}


INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);