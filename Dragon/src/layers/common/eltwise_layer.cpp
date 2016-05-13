#include "layers/common/eltwise_layer.hpp"

template <typename Dtype>
void EltwiseLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK(param.eltwise_param().coeff_size() == 0 || param.eltwise_param().coeff_size() == bottom.size())
		<< "Eltwise Layer takes one coefficient per bottom blob.";
	CHECK(!(param.eltwise_param().operation() == EltwiseParameter_EltwiseOp_PROD&&
		param.eltwise_param().coeff_size()))
		<< "Eltwise layer only takes coefficients for summation.";
	op_ = param.eltwise_param().operation();
	// Blob-wise coefficients for the elementwise operation.
	coeffs_ = vector<Dtype>(bottom.size(), 1);
	if (param.eltwise_param().coeff_size()) {
		for (int i = 0; i < bottom.size(); ++i)
			coeffs_[i] = param.eltwise_param().coeff(i);
	}
	stable_prod_grad_ = param.eltwise_param().stable_prod_grad();
}

template <typename Dtype>
void EltwiseLayer<Dtype>::reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	for (int i = 1; i < bottom.size(); ++i){
		CHECK(bottom[i]->shape() == bottom[0]->shape());
	}

	top[0]->reshapeLike(*bottom[0]);

	// If max operation, we will initialize the vector index part.
	if (param.eltwise_param().operation() == EltwiseParameter_EltwiseOp_MAX && top.size() == 1)
		max_idx_.reshape(bottom[0]->shape());
}

template <typename Dtype>
void EltwiseLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	int* mask = NULL;
	const Dtype* bottom_data_a = NULL;
	const Dtype* bottom_data_b = NULL;
	const int count = top[0]->count();
	Dtype* top_data = top[0]->mutable_cpu_data();
	switch (op_) {
	case EltwiseParameter_EltwiseOp_PROD:
		dragon_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
		for (int i = 2; i < bottom.size(); ++i)
			dragon_mul(count, top_data, bottom[i]->cpu_data(), top_data);
		break;
	case EltwiseParameter_EltwiseOp_SUM:
		dragon_set(count, Dtype(0), top_data);
		// TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
		for (int i = 0; i < bottom.size(); ++i)
			dragon_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
		break;
	case EltwiseParameter_EltwiseOp_MAX:
		// Initialize
		mask = max_idx_.mutable_cpu_data();
		dragon_set(count, -1, mask);
		dragon_set(count, Dtype(-FLT_MAX), top_data);
		// bottom 0 & 1
		bottom_data_a = bottom[0]->cpu_data();
		bottom_data_b = bottom[1]->cpu_data();
		for (int idx = 0; idx < count; ++idx) {
			if (bottom_data_a[idx] > bottom_data_b[idx]) {
				top_data[idx] = bottom_data_a[idx];  // maxval
				mask[idx] = 0;  // maxid
			}
			else {
				top_data[idx] = bottom_data_b[idx];  // maxval
				mask[idx] = 1;  // maxid
			}
		}
		// bottom 2++
		for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
			bottom_data_b = bottom[blob_idx]->cpu_data();
			for (int idx = 0; idx < count; ++idx) {
				if (bottom_data_b[idx] > top_data[idx]) {
					top_data[idx] = bottom_data_b[idx];  // maxval
					mask[idx] = blob_idx;  // maxid
				}
			}
		}
		break;
	default:
		LOG(FATAL) << "Unknown elementwise operation.";
	}
}

template <typename Dtype>
void EltwiseLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom) {
	const int* mask = NULL;
	const int count = top[0]->count();
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	for (int i = 0; i < bottom.size(); ++i) {
		if (data_need_bp[i]) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			switch (op_) {
			case EltwiseParameter_EltwiseOp_PROD:
				if (stable_prod_grad_) {
					bool initialized = false;
					for (int j = 0; j < bottom.size(); ++j) {
						if (i == j) { continue; }
						if (!initialized) {
							dragon_copy(count, bottom_diff, bottom[j]->cpu_data());
							initialized = true;
						}
						else dragon_mul(count, bottom[j]->cpu_data(), bottom_diff, bottom_diff);
					}
				}
				else dragon_div(count, top_data, bottom_data, bottom_diff);
				dragon_mul(count, bottom_diff, top_diff, bottom_diff);
				break;
			case EltwiseParameter_EltwiseOp_SUM:
				if (coeffs_[i] == Dtype(1)) dragon_copy(count, bottom_diff, top_diff);
				else dragon_scale(count, coeffs_[i], top_diff, bottom_diff);
				break;
			case EltwiseParameter_EltwiseOp_MAX:
				mask = max_idx_.cpu_data();
				for (int index = 0; index < count; ++index) {
					Dtype gradient = 0;
					if (mask[index] == i) gradient += top_diff[index];
					bottom_diff[index] = gradient;
				}
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
		}
	}
}

INSTANTIATE_CLASS(EltwiseLayer);