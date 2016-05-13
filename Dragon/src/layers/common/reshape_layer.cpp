#include "layers/common/reshape_layer.hpp"

template <typename Dtype>
void ReshapeLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	infer_axis = -1;
	const BlobShape& param_shape = param.reshape_param().shape();
	const int num_dim = param_shape.dim_size();
	constant_count = 1;
	for (int i = 0; i < num_dim; i++){
		const int dim = param_shape.dim(i);
		if (dim == -1){
			CHECK_EQ(infer_axis, -1) << "ReshapeLayer only accepts one '-1' dim";
			infer_axis = i;
		}
		else if (dim == 0) {
			if (i < 0 || i >= bottom[0]->num_axes())
				LOG(FATAL) << "ReshapeLayer dim-0 with " << i << "th parameter"
				<< " is out of range.";
			constant_count *= bottom[0]->shape(i);
		}
		else constant_count *= dim;
	}
	CHECK_EQ(bottom[0]->count() % constant_count, 0)
		<< "ReshapeLayer can not change the blob's total size.";
}

template <typename Dtype>
void ReshapeLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	int start_axis = param.reshape_param().axis();
	//	'-1' represents the last axis
	start_axis = (start_axis >= 0 ? start_axis : bottom[0]->num_axes() + start_axis);
	CHECK_GE(start_axis, 0)<< "ReshapeLayer start axis" << start_axis << " is out of range.";
	CHECK_LT(start_axis, bottom[0]->num_axes()) << "ReshapeLayer start axis" << start_axis << " is out of range.";
	const int num_axes = param.reshape_param().num_axes();
	CHECK_GE(num_axes, -1) << "ReshapeLayer num axes must >= -1.";
	//	'-1' represents replace till the end
	int end_axis = (num_axes == -1 ? bottom[0]->num_axes() - 1 : start_axis + num_axes);
	CHECK_LT(end_axis, bottom[0]->num_axes()) << "ReshapeLayer end axis" << end_axis << " is out of range.";
	const int num_axes_replaced = end_axis - start_axis + 1;
	const int num_axes_retained = bottom[0]->num_axes() - num_axes_replaced;
	const int num_axes_specified = param.reshape_param().shape().dim_size();
	vector<int> top_shape(num_axes_retained + num_axes_specified);
	int cnt = 0;
	//	retained axes [left]
	for (int i = 0; i < start_axis; i++) top_shape[cnt++] = bottom[0]->shape(i);
	//	replaced axes
	//  compute constant again if bottom.shape had changed (e.g in Faster-RCNN)
	constant_count = 1;
	const BlobShape& param_shape = param.reshape_param().shape();
	for (int i = 0; i < num_axes_specified; i++){
		int dim = param_shape.dim(i);
		if (dim == 0){
			// map to retained axis
			if (i < 0 || i >= bottom[0]->num_axes())
				LOG(FATAL) << "ReshapeLayer dim-0 with " << i << "th parameter"
				<< " is out of range.";
			top_shape[cnt++] = bottom[0]->shape(i);
			constant_count *= bottom[0]->shape(i);
		}
		else{
			if (dim == -1) { cnt++; continue; }
			top_shape[cnt++] = dim;
			constant_count *= dim;
		}
	}
	//	retained axes [right]
	for (int i = end_axis + 1; i<bottom[0]->num_axes(); i++) top_shape[cnt++] = bottom[0]->shape(i);
	//	handle the only '-1' dim
	const int new_infer_axis = start_axis + infer_axis;
	top_shape[new_infer_axis] = bottom[0]->count() / constant_count;
	//	reshape
	top[0]->reshape(top_shape);
	CHECK_EQ(bottom[0]->count(), top[0]->count());
	top[0]->shareData(*bottom[0]);
	top[0]->shareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);