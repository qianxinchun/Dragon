#include "layers/neuron/relu_layer.hpp"

template <typename Dtype>
void ReLULayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int cnt = bottom[0]->count();
	Dtype slope = param.relu_param().negative_slope();
	for (int i = 0; i < cnt; i++)
		top_data[i] = max<Dtype>(bottom_data[i], Dtype(0)) +
		slope*min<Dtype>(bottom_data[i], Dtype(0));
}

template <typename Dtype>
void ReLULayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (data_need_bp[0]){
		Dtype slope = param.relu_param().negative_slope();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int cnt = bottom[0]->count();
		//	bottom_diff = top_diff*1
		//				= slope
		for (int i = 0; i < cnt; i++)
			bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0) + slope*(bottom_data[i] <= 0));
	}
}

INSTANTIATE_CLASS(ReLULayer);