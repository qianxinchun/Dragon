#include "layer_include/neuron_layers.hpp"
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
			bottom_diff[i] = top_diff[i]*((bottom_data[i] > 0) + slope*(bottom_data[i] <= 0));
	}
}

template <typename Dtype>
void DropoutLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	prob = param.dropout_param().prob();
	//	filter threshold(0~UINT_MAX*prob)
	threshold = static_cast<unsigned int>(UINT_MAX*prob);
}

template <typename Dtype>
void DropoutLayer<Dtype>::reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	NeuronLayer<Dtype>::reshape(bottom, top);
	vector<int> shape = bottom[0]->shape();
	rand_vec.reshape(shape);
}

//	recommmend in-place method
template <typename Dtype>
void DropoutLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	unsigned int* mask = rand_vec.mutable_cpu_data();
	const int count = bottom[0]->count();
	if (phase == TRAIN){
		dragon_rng_bernoulli<Dtype>(count, 1 - prob, mask);
		for (int i = 0; i < count; i++)
			top_data[i] = bottom_data[i] * mask[i];
	}
	//	average net
	//	(1-prob)*input=output
	else if (phase == TEST)
		dragon_cpu_axpby<Dtype>(count, Dtype(1) - prob, bottom_data, Dtype(0),top_data);
}

template <typename Dtype>
void DropoutLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (data_need_bp[0]){
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (phase == TRAIN){
			const unsigned int* mask = rand_vec.cpu_data();
			for (int i = 0; i < bottom[0]->count(); i++)
				bottom_diff[i] = top_diff[i] * mask[i];
		}
		else if(phase == TEST){
			NOT_IMPLEMENTED;
		}
	}
}

INSTANTIATE_CLASS(ReLULayer);
INSTANTIATE_CLASS(DropoutLayer);