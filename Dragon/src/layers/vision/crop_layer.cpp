#include "layers/vision/crop_layer.hpp"

template <typename Dtype>
void CropLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const CropParameter crop_param = param.crop_param();
	CHECK_EQ(bottom.size(), 2);
}

template <typename Dtype>
void CropLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){

	/*
		store the offset for each spatial axis
		reshape with the ' ori_shape(i) - offset '
	*/

	//	bottom[0]	----	input blob
	//	bottom[1]	----	referring blob
	//	note that usually we add a large pad(~100) on conv1
	//	input shape is usually larger than referring shape

	const CropParameter crop_param = param.crop_param();
	int num_axes = bottom[0]->num_axes();
	const int start_axis = crop_param.axis();	// default: start_axis=2(first spatial axis)
	offsets = vector<int>(num_axes, 0);
	vector<int> new_shape(bottom[0]->shape());
	
	for (int i = 0; i < num_axes; i++){
		int crop_offset = 0;
		int new_size = bottom[0]->shape(i);
		if (i >= start_axis){
			new_size = bottom[1]->shape(i);
			if (crop_param.offset_size() == 1)
				crop_offset = crop_param.offset(0);	// default: crop_offset=19
			else if (crop_param.offset_size()>1)
				crop_offset = crop_param.offset(i - start_axis);
			CHECK_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i));
		}
		new_shape[i] = new_size;
		offsets[i] = crop_offset;
	}
	top[0]->reshape(new_shape);
}

template <typename Dtype>
void CropLayer<Dtype>::copy(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top,
	const vector<int>& offsets, vector<int> idxs, int cur_dim, const Dtype* src_data,
	Dtype* dest_data, bool is_forward){

	//	recursive-term
	if (cur_dim + 1 < top[0]->num_axes()){
		for (int i = 0; i < top[0]->shape(cur_dim); i++){
			//	store the pixel-idx of the current spatial axis
			idxs[cur_dim] = i;
			//	recursive for spatial axis
			copy(bottom, top, offsets, idxs, cur_dim + 1, src_data, dest_data, is_forward);
		}
	}
	//	terminal-term
	//	perform a linear-mem copy at the last spatial axis
	else{
		for (int i = 0; i < top[0]->shape(cur_dim); i++){
			//	offset_baseline = n*c*h
			vector<int> idx_off(cur_dim + 1, 0);
			for (int j = 0; j < cur_dim; j++) idx_off[j] = idxs[j] + offsets[j];
			idx_off[cur_dim] = offsets[cur_dim]; // only do offset
			if (is_forward){
				//	src: bottom_data£¬dest: top_data
				dragon_copy(top[0]->shape(cur_dim), dest_data + top[0]->offset(idxs),
					src_data + bottom[0]->offset(idx_off));
			}else{
				//	src: top_diff£¬dest: bottom_diff
				dragon_copy(top[0]->shape(cur_dim), dest_data + bottom[0]->offset(idx_off),
					src_data + top[0]->offset(idxs));
			}
		}
	}
}

template <typename Dtype>
void CropLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	vector<int> idxs(top[0]->num_axes(), 0);
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	copy(bottom, top, offsets, idxs, 0, bottom_data, top_data, true);
}

template <typename Dtype>
void CropLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (!data_need_bp[0]) return;
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	//	must clear the last diff due to the different shape according mini-batches
	dragon_set(bottom[0]->count(), Dtype(0), bottom_diff);
	vector<int> idxs(top[0]->num_axes(), 0);
	copy(bottom, top, offsets, idxs, 0, top_diff, bottom_diff, false);
}

INSTANTIATE_CLASS(CropLayer);