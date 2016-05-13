#include "layers/common/softmax_layer.hpp"

template <typename Dtype>
__global__ void MaxClassKernel(const int outer_num, const int classes, const int inner_num,
	const Dtype* bottom_data, Dtype* scale_data){
	CUDA_KERNEL_LOOP(idx, outer_num*inner_num){
		int o_idx = idx / inner_num;
		int i_idx = idx% inner_num;
		Dtype max_val = -FLT_MAX;
		for (int c = 0; c < classes; c++)
			max_val = max(bottom_data[(o_idx*classes + c)*inner_num + i_idx], max_val);
		scale_data[idx] = max_val;
	}
}

template <typename Dtype>
__global__ void SubtractKernel(const int count, const int classes, const int inner_num,
	const Dtype* scale_data, Dtype* top_data){
	//	count=outer_num*classes*inner_num
	CUDA_KERNEL_LOOP(idx, count){
		int o_idx = idx / inner_num / classes;
		int i_idx = idx% inner_num;
		//	ignore classes
		//	note that scale_data shapeis [outer_num,inner_num]
		top_data[idx] -= scale_data[o_idx*inner_num + i_idx];
	}
}

template <typename Dtype>
__global__ void ExpKernel(const int count, Dtype* top_data){
	CUDA_KERNEL_LOOP(idx, count){
		top_data[idx] = exp(top_data[idx]);
	}
}

template <typename Dtype>
__global__ void SumClassKernel(const int outer_num, const int classes, const int inner_num,
	const Dtype* top_data, Dtype* scale_data){
	CUDA_KERNEL_LOOP(idx, outer_num*inner_num){
		int o_idx = idx / inner_num;
		int i_idx = idx% inner_num;
		Dtype sum = 0;
		for (int c = 0; c < classes; c++)
			sum += top_data[(o_idx*classes + c)*inner_num + i_idx];
		scale_data[idx] = sum;
	}
}

template <typename Dtype>
__global__ void DivKernel(const int count, const int classes, const int inner_num,
	const Dtype* scale_data, Dtype* top_data){
	//	count=outer_num*classes*inner_num
	CUDA_KERNEL_LOOP(idx, count){
		int o_idx = idx / inner_num / classes;
		int i_idx = idx% inner_num;
		//	ignore classes
		//	note that scale_data shapeis [outer_num,inner_num]
		top_data[idx] /= scale_data[o_idx*inner_num + i_idx];
	}
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype *bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* scale_data = scale.mutable_gpu_data();
	//	num_class
	const int classes = bottom[0]->shape(axis);
	const int count = bottom[0]->count();
	//	normally the dim equal to classes
	//	spacially if we do not connect a inner product layer before
	//	we may get a 4D input and dim=classes*height*width
	dragon_gpu_copy(count, top_data, bottom_data);

	MaxClassKernel<Dtype> << <GET_BLOCKS(inner_num*outer_num), CUDA_NUM_THREADS >> >(
		outer_num, classes, inner_num, bottom_data, scale_data);
	SubtractKernel<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, classes, inner_num, scale_data, top_data);
	ExpKernel<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(count, top_data);
	SumClassKernel<Dtype> << <GET_BLOCKS(inner_num*outer_num), CUDA_NUM_THREADS >> >(
		outer_num, classes, inner_num, top_data, scale_data);
	DivKernel<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, classes, inner_num, scale_data, top_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void DotKernel(const int outer_num, const int classes, const int inner_num,
	const Dtype* top_diff, const Dtype* top_data, Dtype* scale_data){
	CUDA_KERNEL_LOOP(idx, outer_num*inner_num){
		int o_idx = idx / inner_num;
		int i_idx = idx% inner_num;
		Dtype dot = 0;
		for (int c = 0; c < classes; c++)
			dot += (top_data[(o_idx*classes + c)*inner_num + i_idx]
			* top_diff[(o_idx*classes + c)*inner_num + i_idx]);
		scale_data[idx] = dot;
	}
}


template <typename Dtype>
void SoftmaxLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* top_data = top[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* scale_data = scale.mutable_gpu_data();
	int classes = top[0]->shape(axis);
	int count = top[0]->count() / outer_num;
	dragon_gpu_copy(count, bottom_diff, top_diff);
	//	softmax and loss layer is splitted in Caffe
	//	please read https://www.zhihu.com/question/28927103 before
	//	for each example
	DotKernel<Dtype> << <GET_BLOCKS(inner_num*outer_num), CUDA_NUM_THREADS >> >(
		outer_num, classes, inner_num, top_diff, top_data, scale_data);
	SubtractKernel<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, classes, inner_num, scale_data, bottom_diff);

	dragon_gpu_mul(count, bottom_diff, top_data, bottom_diff);
	CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);