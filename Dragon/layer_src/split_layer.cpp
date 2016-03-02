#include "layer_include/common_layers.hpp"
template <typename Dtype>
void SplitLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	count = bottom[0]->count();
	for (int i = 0; i < top.size(); i++){
		top[i]->reshapeLike(*bottom[0]);
		CHECK_EQ(count, top[i]->count());
	}
}


//	forward just share the data
template <typename Dtype>
void SplitLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	for (int i = 0; i < top.size(); i++) top[i]->shareData(*bottom[0]);
}

//	backward just sum up all splited diff
template <typename Dtype>
void SplitLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	if (!data_need_bp[0]) return;
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	for (int i = 0; i < top.size(); i++){
		const Dtype* top_diff = top[i]->cpu_diff();
		dragon_axpy<Dtype>(count, Dtype(1.0), top_diff, bottom_diff);
	}
}

template <typename Dtype>
void SplitLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	for (int i = 0; i < top.size(); i++) top[i]->shareData(*bottom[0]);
}

template <typename Dtype>
void SplitLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	if (!data_need_bp[0]) return;
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	for (int i = 0; i < top.size(); i++){
		const Dtype* top_diff = top[i]->gpu_diff();
		dragon_gpu_axpy<Dtype>(count, Dtype(1.0), top_diff, bottom_diff);
	}
}

INSTANTIATE_CLASS(SplitLayer);