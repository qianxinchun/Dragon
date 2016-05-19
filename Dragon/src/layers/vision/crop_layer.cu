#include "layers/vision/crop_layer.hpp"

//	a linear-mem copy kernel for the last 'LEVEL' spatial axes
//	we re-implement it which is much efficient than blvc-caffe version

template <typename Dtype>
__global__ void	CopyKernel(const int n, const int* src_dims,const int* dest_dims,const Dtype* src, Dtype* dest){
	CUDA_KERNEL_LOOP(idx, n){
		int w = idx%dest_dims[LEVEL - 2];
		int h = (idx / dest_dims[LEVEL - 2]) % dest_dims[LEVEL - 3];
		int c = (idx / dest_dims[LEVEL - 2] / dest_dims[LEVEL - 3]);
		int dest_idx = (c*dest_dims[LEVEL - 3] + h)*dest_dims[LEVEL - 2] + w;
		int src_idx = (c*src_dims[LEVEL - 3] + h)*src_dims[LEVEL - 2] + w;
		dest[dest_idx] = src[src_idx];
	}
}


template <typename Dtype>
void CropLayer<Dtype>::copy_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top,
	const vector<int>& offsets, vector<int> idxs, int cur_dim, const Dtype* src_data,
	Dtype* dest_data, bool is_forward){

	//	recursive-term
	if (cur_dim + LEVEL < top[0]->num_axes()){
		for (int i = 0; i < top[0]->shape(cur_dim); i++){
			//	store the pixel-idx of the current spatial axis
			idxs[cur_dim] = i;
			//	recursive for spatial axis
			copy_gpu(bottom, top, offsets, idxs, cur_dim + 1, src_data, dest_data, is_forward);
		}
	}
	//	terminal-term
	//	perform a linear-mem copy for the last 'LEVEL' spatial axes
	//	you can choose Level2 or Level3 parallel algorithms
	else{
		int outer_dim = 1;
		for (int i = 0; i < LEVEL; i++) outer_dim *= top[0]->shape(cur_dim + i);
		vector<int> idx_off(cur_dim + LEVEL, 0);
		for (int j = 0; j < cur_dim; j++) idx_off[j] = idxs[j] + offsets[j];
		for (int j = 0; j < LEVEL; j++) idx_off[cur_dim+j] = offsets[cur_dim + j];;
		for (int j = LEVEL-2; j >=0; j--){
			int* src = src_dims_blob.mutable_cpu_data();
			int* dest = dest_dims_blob.mutable_cpu_data();
			dest[j] = top[0]->shape(cur_dim + j+1);
			src[j] = bottom[0]->shape(cur_dim + j+1);
		}
		const int* src_dims = src_dims_blob.gpu_data();
		const int* dest_dims = dest_dims_blob.gpu_data();
		//
		if (is_forward){
			const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(idx_off);
			Dtype* top_data = top[0]->mutable_gpu_data() + top[0]->offset(idxs);
			CopyKernel<Dtype> << <GET_BLOCKS(outer_dim), CUDA_NUM_THREADS >> >(
				outer_dim, src_dims,dest_dims,bottom_data, top_data);
		}else{
			const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(idxs);
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(idx_off);
			CopyKernel<Dtype> << <GET_BLOCKS(outer_dim), CUDA_NUM_THREADS >> >(
				outer_dim,  dest_dims, src_dims, top_diff, bottom_diff);
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