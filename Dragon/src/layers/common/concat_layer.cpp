#include "layers/common/concat_layer.hpp"

template <typename Dtype>
void ConcatLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const int num_axes = bottom[0]->num_axes();
	axis = 1;
	vector<int> top_shape = bottom[0]->shape();
	batch_size = bottom[0]->count(0, axis);
	input_size = bottom[0]->count(axis + 1);
	int count_sum = bottom[0]->count();
	// CHECK all axes except axis(channels)
	for (int i = 1; i < bottom.size(); i++){
		CHECK_EQ(num_axes, bottom[i]->num_axes())
			<< "All inputs must have the same axes.";
		for (int j = 0; j < num_axes; j++){
			if (j == axis) continue;
			CHECK_EQ(top_shape[j], bottom[i]->shape(j))
				<< "All inputs must have the same shape, except axis(channels).";
		}
		count_sum += bottom[i]->count();
		top_shape[axis] += bottom[i]->shape(axis);
	}
	top[0]->reshape(top_shape);
	CHECK_EQ(count_sum, top[0]->count());
	//	optimize if specify only one bottom blob
	if (bottom.size() == 1){
		top[0]->shareData(*bottom[0]);
		top[0]->shareDiff(*bottom[0]);
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	if (bottom.size() == 1) return;
	Dtype* top_data = top[0]->mutable_cpu_data();
	int channels_offset = 0, num_output = top[0]->shape(axis);
	for (int i = 0; i < bottom.size(); i++){
		const Dtype* bottom_data = bottom[i]->cpu_data();
		const int channels = bottom[i]->shape(axis);
		//	concat through the whole batch
		for (int n = 0; n < batch_size; n++){
			int copy_size = channels*input_size;
			int bottom_offset = n*copy_size;
			int top_offset = (n*num_output + channels_offset)*input_size;
			dragon_copy<Dtype>(copy_size, top_data + top_offset, bottom_data + bottom_offset);
		}
		channels_offset += channels;
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp,
	const vector<Blob<Dtype>*> &bottom){
	if (bottom.size() == 1) return;
	const Dtype* top_diff = top[0]->cpu_diff();
	int channels_offset = 0, num_output = top[0]->shape(axis);
	for (int i = 0; i < bottom.size(); i++){
		const int channels = bottom[i]->shape(axis);
		if (data_need_bp[i]){
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			for (int n = 0; n < batch_size; n++){
				int copy_size = channels*input_size;
				int bottom_offset = n*copy_size;
				int top_offset = (n*num_output + channels_offset)*input_size;
				dragon_copy<Dtype>(copy_size, bottom_diff + bottom_offset, top_diff + top_offset);
			}
		}
		channels_offset += channels;
	}
}

INSTANTIATE_CLASS(ConcatLayer);