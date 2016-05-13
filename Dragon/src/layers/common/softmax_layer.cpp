#include "layers/common/softmax_layer.hpp"

template <typename Dtype>
void SoftmaxLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	SoftmaxParameter softmax_param = param.softmax_param();
	axis = bottom[0]->canonicalAxisIndex(softmax_param.axis());
	top[0]->reshapeLike(*bottom[0]);
	//	num of classes
	vector<int> mult_dims(1, bottom[0]->shape(axis));
	sum_multiplier.reshape(mult_dims);
	dragon_set(sum_multiplier.count(), (Dtype)1.0, sum_multiplier.mutable_cpu_data());
	//	see loss_layer.cpp/SoftmaxWithLossLayer<Dtype>::reshape()
	outer_num = bottom[0]->count(0, axis);
	inner_num = bottom[0]->count(axis + 1);
	//	(batch,classes) -> (batch)
	//	we use scale to store temporary component for an example
	vector<int> scale_dims = bottom[0]->shape();
	scale_dims[axis] = 1;
	scale.reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	const Dtype *bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* scale_data = scale.mutable_cpu_data();
	//	num_class
	int classes = bottom[0]->shape(axis);
	//	normally the dim equal to classes
	//	spacially if we do not connect a inner product layer before
	//	we may get a 4D input and dim=classes*height*width
	int dim = bottom[0]->count() / outer_num;
	dragon_copy(bottom[0]->count(), top_data, bottom_data);
	//	for(each example)
	for (int i = 0; i < outer_num; i++){
		//	copy the first class's values of this example to stuff in the scale
		dragon_copy(inner_num, scale_data, bottom_data + i*dim);
		//	find the max values of all classes and replace them in the scale
		for (int j = 0; j < classes; j++){
			for (int k = 0; k < inner_num; k++)
				scale_data[k] = max(scale_data[k], bottom_data[i*dim + j*inner_num + k]);
		}
		//	subtract the max values for each classes in the scale
		//	note that it is additional operation in Softmax which relieve numerical issues
		//	implemented by Caffe and we do not know why do this exclusively
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, classes, inner_num, 1,
			(Dtype)-1.0, sum_multiplier.cpu_data(), scale_data, (Dtype)1.0, top_data);
		//	exp all (Wx+b) term
		dragon_exp<Dtype>(dim, top_data, top_data);
		//	Transpose and sum up classes_sum_exp_term as Softmax-Denominator in the scale
		dragon_cpu_gemv(CblasTrans, classes, inner_num,
			(Dtype)1.0, top_data, sum_multiplier.cpu_data(), (Dtype)0, scale_data);
		//	divide a Softmax-Denominator for each classes
		//	and the Softmax-Numerator is e^(Wx+b) for each classes
		//	after that we get the full and original Softmax-Matrix
		for (int j = 0; j < classes; j++){
			dragon_div(inner_num, top_data, scale_data, top_data);
			//cout << *top_data << endl;
			//	shapes(e.g. H/W) offset and normally inner_num=1
			top_data += inner_num;
		}
	}
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* scale_data = scale.mutable_cpu_data();
	int classes = top[0]->shape(axis);
	int dim = top[0]->count() / outer_num;
	dragon_copy(top[0]->count(), bottom_diff, top_diff);
	//	softmax and loss layer is splitted in Caffe
	//	please read https://www.zhihu.com/question/28927103 before
	//	for each example
	for (int i = 0; i < outer_num; i++){
		//	compute [dl/da]*a use vec_dot
		//  [dl/da_(i)]*a_(i) = top_diff_(i)*top_data_(i) 
		for (int k = 0; k < inner_num; k++)
			scale_data[k] = dragon_cpu_strided_dot<Dtype>(classes, bottom_diff + i*dim + k, inner_num,
			top_data + i*dim + k, inner_num);
		//	repeate the scale value into a scale vector
		//	top_diff_vec -= repeate_vec
		dragon_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, classes, inner_num, 1,
			(Dtype)-1.0, sum_multiplier.cpu_data(), scale_data, (Dtype)1.0, bottom_diff + i*dim);
	}
	//	mul a_(i) for each diff
	//	after that the bottom_diff is equal to < 1(y=i)-p(y=i|x,theta) >
	//	the form which combine the softmax and loss layer look much simple
	dragon_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_CLASS(SoftmaxLayer);