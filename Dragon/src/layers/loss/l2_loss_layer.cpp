#include "layers/loss/l2_loss_layer.hpp"

template <typename Dtype>
void L2LossLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	LossLayer<Dtype>::reshape(bottom, top);
	diff.reshapeLike(*bottom[0]);
}

template <typename Dtype>
void L2LossLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	int count = bottom[0]->count();
	//	compute diff
	dragon_sub < Dtype>(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff.mutable_cpu_data());
	//	compute diff^2
	Dtype dot = dragon_cpu_dot<Dtype>(count, diff.cpu_data(), diff.cpu_data());
	//	compute LMS
	Dtype loss = dot / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L2LossLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	for (int i = 0; i < 2; i++){
		if (data_need_bp[i]){
			const Dtype sign = (i == 0) ? 1 : -1;
			//	sign*loss_weight/batch_size
			const Dtype alpha = sign*top[0]->cpu_diff()[0] / bottom[i]->num();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			dragon_cpu_axpby<Dtype>(bottom[i]->count(), alpha, diff.cpu_data(), Dtype(0), bottom_diff);
		}
	}
}

template <typename Dtype>
void L2LossLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	int count = bottom[0]->count();
	//	compute diff
	dragon_gpu_sub <Dtype>(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff.mutable_gpu_data());
	//	compute diff^2
	Dtype dot = dragon_gpu_dot<Dtype>(count, diff.gpu_data(), diff.gpu_data());
	//	compute LMS
	Dtype loss = dot / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L2LossLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	for (int i = 0; i < 2; i++){
		if (data_need_bp[i]){
			const Dtype sign = (i == 0) ? 1 : -1;
			//	sign*loss_weight/batch_size
			const Dtype alpha = sign*top[0]->cpu_diff()[0] / bottom[i]->num();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			dragon_gpu_axpby<Dtype>(bottom[i]->count(), alpha, diff.gpu_data(), Dtype(0), bottom_diff);
		}
	}
}

INSTANTIATE_CLASS(L2LossLayer);