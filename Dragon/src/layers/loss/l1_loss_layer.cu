#include "layers/loss/l1_loss_layer.hpp"

template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out, const Dtype sigma2){
	CUDA_KERNEL_LOOP(idx, n){
		Dtype val = in[idx];
		Dtype abs_val = abs(val);
		if (abs_val < 1.0 / sigma2) out[idx] = 0.5*val*val*sigma2;
		else out[idx] = abs_val - 0.5 / sigma2;
	}
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	int count = bottom[0]->count();
	dragon_gpu_sub<Dtype>(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff.mutable_gpu_data());
	if (has_weights)
		//	apply inside weights (0 or 1)
		dragon_gpu_mul<Dtype>(count, bottom[2]->gpu_data(), diff.gpu_data(), diff.mutable_gpu_data());
	SmoothL1Forward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, diff.gpu_data(), errors.mutable_gpu_data(), sigma2);
	CUDA_POST_KERNEL_CHECK;
	if (has_weights)
		//	apply outside weights (normalize)
		dragon_gpu_mul<Dtype>(count, bottom[3]->gpu_data(), errors.gpu_data(), errors.mutable_gpu_data());
	Dtype loss = dragon_gpu_asum<Dtype>(count, errors.gpu_data());
	top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out, const Dtype sigma2){
	CUDA_KERNEL_LOOP(idx, n){
		Dtype val = in[idx];
		Dtype abs_val = abs(val);
		if (abs_val < 1.0 / sigma2) out[idx] = val*sigma2;
		//	val>0: 1 | val=0: 0 | val<0: -1
		else out[idx] = (val > Dtype(0)) - (val < Dtype(0));
	}
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	int count = diff.count();
	SmoothL1Backward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
		count, diff.gpu_data(), diff.mutable_gpu_data(), sigma2);
	CUDA_POST_KERNEL_CHECK;
	for (int i = 0; i < 2; i++){
		if (data_need_bp[i]){
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign*top[0]->cpu_diff()[0] / bottom[i]->num();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			dragon_gpu_axpby<Dtype>(count, alpha, diff.gpu_data(), Dtype(0), bottom_diff);
			if (has_weights){
				//	inside
				dragon_gpu_mul<Dtype>(count, bottom[2]->gpu_data(), bottom[i]->gpu_diff(), bottom_diff);
				//	outside
				dragon_gpu_mul<Dtype>(count, bottom[3]->gpu_data(), bottom[i]->gpu_diff(), bottom_diff);
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);