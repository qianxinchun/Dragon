#include "layers/neuron/power_layer.hpp"

template <typename Dtype>
void PowerLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::layerSetup(bottom, top);
	power_ = param.power_param().power();
	scale_ = param.power_param().scale();
	shift_ = param.power_param().shift();
	diff_scale_ = power_  * scale_;
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
void PowerLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	// Special case where we can ignore the input: scale or power is 0.
	if (diff_scale_ == Dtype(0)) {
		Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
		dragon_set(count, value, top_data);
		return;
	}
	const Dtype* bottom_data = bottom[0]->cpu_data();
	dragon_copy(count, top_data, bottom_data);
	if (scale_ != Dtype(1)) dragon_scal(count, scale_, top_data);
	if (shift_ != Dtype(0)) dragon_add_scalar(count, shift_, top_data);
	if (power_ != Dtype(1)) dragon_powx(count, top_data, power_, top_data);

}

template <typename Dtype>
void PowerLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& data_need_bp,
	const vector<Blob<Dtype>*>& bottom) {
	if (data_need_bp[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		const Dtype* top_diff = top[0]->cpu_diff();
		if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) dragon_set(count, diff_scale_, bottom_diff);
		else {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			// Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
			//               = diff_scale * y / (shift + scale * x)
			if (power_ == Dtype(2)) {
				// Special case for y = (shift + scale * x)^2
				//     -> dy/dx = 2 * scale * (shift + scale * x)
				//              = diff_scale * shift + diff_scale * scale * x
				dragon_cpu_axpby(count, diff_scale_ * scale_, bottom_data, Dtype(0), bottom_diff);
				if (shift_ != Dtype(0)) dragon_add_scalar(count, diff_scale_ * shift_, bottom_diff);
			}
			else if (shift_ == Dtype(0)) {
				// Special case for y = (scale * x)^power
				//     -> dy/dx = scale * power * (scale * x)^(power - 1)
				//              = scale * power * (scale * x)^power * (scale * x)^(-1)
				//              = power * y / x
				const Dtype* top_data = top[0]->cpu_data();
				dragon_div(count, top_data, bottom_data, bottom_diff);
				dragon_scal(count, power_, bottom_diff);
			}
			else {
				dragon_copy(count, bottom_diff, bottom_data);
				if (scale_ != Dtype(1)) dragon_scal(count, scale_, bottom_diff);
				if (shift_ != Dtype(0)) dragon_add_scalar(count, shift_, bottom_diff);
				const Dtype* top_data = top[0]->cpu_data();
				dragon_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
				if (diff_scale_ != Dtype(1)) dragon_scal(count, diff_scale_, bottom_diff);
			}
		}
		if (diff_scale_ != Dtype(0)) dragon_mul(count, top_diff, bottom_diff, bottom_diff);
	}
}

INSTANTIATE_CLASS(PowerLayer);