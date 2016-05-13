#include "layers/common/eltwise_layer.hpp"


template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
	const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data, int* mask) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype maxval = -FLT_MAX;
		int maxidx = -1;
		if (bottom_data_a[index] > bottom_data_b[index]) {
			// only update for very first bottom_data blob (blob_idx == 0)
			if (blob_idx == 0) {
				maxval = bottom_data_a[index];
				top_data[index] = maxval;
				maxidx = blob_idx;
				mask[index] = maxidx;
			}
		}
		else {
			maxval = bottom_data_b[index];
			top_data[index] = maxval;
			maxidx = blob_idx + 1;
			mask[index] = maxidx;
		}
	}
}

template <typename Dtype>
void EltwiseLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	int* mask = NULL;
	const int count = top[0]->count();
	Dtype* top_data = top[0]->mutable_gpu_data();
	switch (op_) {
	case EltwiseParameter_EltwiseOp_PROD:
		dragon_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
		for (int i = 2; i < bottom.size(); ++i)
			dragon_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
		break;
	case EltwiseParameter_EltwiseOp_SUM:
		dragon_gpu_set(count, Dtype(0.), top_data);
		// TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
		for (int i = 0; i < bottom.size(); ++i)
			dragon_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
		break;
	case EltwiseParameter_EltwiseOp_MAX:
		mask = max_idx_.mutable_gpu_data();
		// NOLINT_NEXT_LINE(whitespace/operators)
		MaxForward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
			count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
		for (int i = 2; i < bottom.size(); ++i) {
			// NOLINT_NEXT_LINE(whitespace/operators)
			MaxForward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
				count, top_data, bottom[i]->gpu_data(), i - 1, top_data, mask);
		}
		break;
	default:
		LOG(FATAL) << "Unknown elementwise operation.";
	}
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
	const int blob_idx, const int* mask, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype gradient = 0;
		if (mask[index] == blob_idx) {
			gradient += top_diff[index];
		}
		bottom_diff[index] = gradient;
	}
}

template <typename Dtype>
void EltwiseLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom) {
	const int* mask = NULL;
	const int count = top[0]->count();
	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	for (int i = 0; i < bottom.size(); ++i) {
		if (data_need_bp[i]) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			switch (op_) {
			case EltwiseParameter_EltwiseOp_PROD:
				if (stable_prod_grad_) {
					bool initialized = false;
					for (int j = 0; j < bottom.size(); ++j) {
						if (i == j) { continue; }
						if (!initialized) {
							dragon_gpu_copy(count, bottom_diff, bottom[j]->gpu_data());
							initialized = true;
						}
						else dragon_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff, bottom_diff);
					}
				}
				else dragon_gpu_div(count, top_data, bottom_data, bottom_diff);
				dragon_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
				break;
			case EltwiseParameter_EltwiseOp_SUM:
				if (coeffs_[i] == Dtype(1.)) dragon_gpu_copy(count, bottom_diff, top_diff);
				else dragon_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
				break;
			case EltwiseParameter_EltwiseOp_MAX:
				mask = max_idx_.gpu_data();
				MaxBackward<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(count, top_diff, i, mask, bottom_diff);
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);